"""Manual multi-hop validation of the agentic loop against Lewis et al. (2020) RAG paper.

Reuses the Milestone 03 corpus/collection and query set from ``validate_retrieval.py`` and, per
query, records how many retrieve/grade passes ran, every rewrite, every grader verdict, and
whether the final citations are grounded in retrieved chunks.
"""

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx
from qdrant_client import AsyncQdrantClient
from validate_retrieval import (
    _DEFAULT_COLLECTION,
    _DEFAULT_RERANKER_MODEL,
    _DEFAULT_SOURCE_URI,
    ValidationQuery,
    _create_langfuse_client,
    _ensure_validation_corpus,
    _load_queries,
)

from agentic_rag_copilot.adapters.embedding import OllamaEmbedder
from agentic_rag_copilot.adapters.llm import OllamaLLM
from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker
from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.agent import (
    AgenticLoopService,
    AgentSettings,
    AnswerSynthesizer,
    QueryRewriter,
    RelevanceGrader,
)
from agentic_rag_copilot.domain.models import AgentState, AskRequest, GradeResult
from agentic_rag_copilot.ingestion import IngestionSettings
from agentic_rag_copilot.logging import get_logger, setup_logging
from agentic_rag_copilot.services import RetrievalService

logger = get_logger(__name__)

_DEFAULT_TOP_K: Final[int] = 5
_ANSWER_SNIPPET_LENGTH: Final[int] = 160
_EXIT_OK: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1


@dataclass
class _Trace:
    """Grader verdicts and rewrites observed while answering one query."""

    grades: list[GradeResult] = field(default_factory=list[GradeResult])
    rewrites: list[str] = field(default_factory=list[str])


class _TracingGrader(RelevanceGrader):
    def __init__(self, llm: OllamaLLM, trace: _Trace) -> None:
        super().__init__(llm)
        self.trace = trace

    async def grade(self, state: AgentState) -> GradeResult:
        result = await super().grade(state)
        self.trace.grades.append(result)
        return result


class _TracingRewriter(QueryRewriter):
    def __init__(self, llm: OllamaLLM, trace: _Trace) -> None:
        super().__init__(llm)
        self.trace = trace

    async def rewrite(self, state: AgentState, reasoning: str | None = None) -> str:
        rewritten = await super().rewrite(state, reasoning=reasoning)
        self.trace.rewrites.append(rewritten)
        return rewritten


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _format_query_section(
    vq: ValidationQuery, trace: _Trace, iterations: int, answer: str, citation_ids: list[str]
) -> str:
    verdicts = " -> ".join("✅" if g.is_relevant else "❌" for g in trace.grades)
    gated = trace.grades[-1].is_relevant if trace.grades else False
    lines = [
        f"### ({vq.category}) {vq.text}",
        "",
        f"- **Passes (`iterations_taken`)**: {iterations} | **Rewrites**: {len(trace.rewrites)}",
        f"- **Grader verdicts**: {verdicts} | **Generation gated by sufficient grade**: {gated}",
        f"- **Citations**: {len(citation_ids)} ({', '.join(c[:8] for c in citation_ids) or '-'})",
        f"- **Answer**: {_escape(answer[:_ANSWER_SNIPPET_LENGTH])}",
        "",
        "| Pass | Query | Verdict | Grader reasoning |",
        "|---:|---|---|---|",
    ]
    queries = [vq.text, *trace.rewrites]
    for idx, grade in enumerate(trace.grades, start=1):
        query = queries[idx - 1] if idx - 1 < len(queries) else "?"
        lines.append(
            f"| {idx} | {_escape(query)} | {'✅' if grade.is_relevant else '❌'} | "
            f"{_escape(grade.reasoning)} |"
        )
    lines.append("")
    return "\n".join(lines)


async def _run(
    ingestion: IngestionSettings,
    agent: AgentSettings,
    queries: list[ValidationQuery],
    output_path: Path | None,
    *,
    collection_name: str,
    source_uri: str,
    top_k: int,
    reingest: bool,
) -> int:
    qdrant = AsyncQdrantClient(url=ingestion.qdrant_url)
    sections = [
        "# Agentic Loop Validation Output",
        "",
        f"Source: `{source_uri}` | Collection: `{collection_name}` | LLM: `{agent.llm_model}` | "
        f"`max_loop_count={agent.max_loop_count}` | `top_k={top_k}`",
        "",
    ]
    try:
        async with (
            httpx.AsyncClient(
                base_url=ingestion.ollama_url, timeout=ingestion.embedding_timeout_seconds
            ) as embed_client,
            httpx.AsyncClient(
                base_url=agent.ollama_url, timeout=agent.llm_timeout_seconds
            ) as llm_client,
        ):
            embedder = OllamaEmbedder(
                embed_client,
                model=ingestion.embedding_model,
                batch_size=ingestion.embedding_batch_size,
            )
            vector_store = QdrantVectorStore(qdrant, embedder, collection_name)
            await _ensure_validation_corpus(
                qdrant,
                embedder,
                vector_store,
                ingestion,
                collection_name,
                source_uri,
                reingest=reingest,
            )
            langfuse = _create_langfuse_client()
            retrieval = RetrievalService(
                embedder=embedder,
                vector_store=vector_store,
                reranker=FastEmbedReranker(model_name=_DEFAULT_RERANKER_MODEL),
                langfuse_client=langfuse,
            )
            llm = OllamaLLM(llm_client, model=agent.llm_model)

            for idx, vq in enumerate(queries, start=1):
                logger.info(f"[{idx}/{len(queries)}] ({vq.category}) {vq.text}")
                trace = _Trace()
                loop = AgenticLoopService(
                    retrieval_service=retrieval,
                    grader=_TracingGrader(llm, trace),
                    rewriter=_TracingRewriter(llm, trace),
                    synthesizer=AnswerSynthesizer(llm),
                    settings=agent,
                    langfuse_client=langfuse,
                )
                try:
                    response = await loop.ask(AskRequest(query=vq.text, top_k=top_k))
                except ValueError as exc:
                    logger.error(f"Query failed: {exc}")
                    sections.append(f"### ({vq.category}) {vq.text}\n\n- **FAILED**: {exc}\n")
                    continue
                section = _format_query_section(
                    vq,
                    trace,
                    response.iterations_taken,
                    response.answer,
                    [c.chunk_id for c in response.citations],
                )
                sections.append(section)
                logger.info(f"\n{section}")
            langfuse.flush()
    except Exception as exc:
        logger.error(f"Validation failed: {exc}")
        return _EXIT_FAILURE
    finally:
        await qdrant.close()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(sections), encoding="utf-8")
        logger.info(f"Wrote agentic loop validation report to {output_path}")
    return _EXIT_OK


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_agentic_loop",
        description="Run the agentic loop over the Milestone 03 validation queries.",
    )
    parser.add_argument("--source", default=_DEFAULT_SOURCE_URI)
    parser.add_argument("--collection", default=_DEFAULT_COLLECTION)
    parser.add_argument("--reingest", action="store_true")
    parser.add_argument("--queries-file", type=Path, default=None)
    parser.add_argument(
        "--category",
        default=None,
        help="Only run queries from this category (e.g. multi_hop).",
    )
    parser.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K)
    parser.add_argument(
        "--max-loop-count",
        type=int,
        default=None,
        help="Override RAG_MAX_LOOP_COUNT for this run.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the agentic loop validation.

    Args:
        argv: Optional command line arguments.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    setup_logging()
    args = _parse_args(argv)
    agent = AgentSettings()
    if args.max_loop_count is not None:
        agent = agent.model_copy(update={"max_loop_count": args.max_loop_count})
    queries = _load_queries(args.queries_file)
    if args.category is not None:
        queries = [q for q in queries if q.category == args.category]
    return asyncio.run(
        _run(
            IngestionSettings(),
            agent,
            queries,
            args.output,
            collection_name=args.collection,
            source_uri=args.source,
            top_k=args.top_k,
            reingest=args.reingest,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
