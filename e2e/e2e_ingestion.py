"""Standalone ingestion + retrieval smoke test for a multi-document corpus.

Ingests any number of PDFs/HTML pages into a dedicated Qdrant collection and runs a batch
of queries (hybrid search + cross-encoder reranking), reporting per-query latency and which
source document each result came from.
"""

import argparse
import asyncio
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx
from dotenv import load_dotenv
from langfuse import Langfuse
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.embedding import OllamaEmbedder
from agentic_rag_copilot.adapters.ingestion import (
    LangchainTextChunker,
    PyMuPDFLoader,
    TrafilaturaHTMLLoader,
)
from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker
from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.domain.models import Chunk, IngestRequest, Query, SourceType
from agentic_rag_copilot.ingestion import (
    IngestionError,
    IngestionPipeline,
    IngestionSettings,
    detect_source_type,
)
from agentic_rag_copilot.logging import get_logger, setup_logging
from agentic_rag_copilot.services import RetrievalService

_logger = get_logger(__name__)

_DEFAULT_COLLECTION: Final[str] = "e2e_ingestion"
_DEFAULT_RERANKER_MODEL: Final[str] = "jinaai/jina-reranker-v2-base-multilingual"
_DEFAULT_INITIAL_TOP_K: Final[int] = 30
_DEFAULT_FINAL_TOP_K: Final[int] = 6
_SNIPPET_LENGTH: Final[int] = 100
_EXIT_OK: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1


@dataclass(frozen=True)
class QueryCase:
    """A single query to evaluate, optionally labeled with a category."""

    text: str
    category: str = "custom"


@dataclass
class IngestionStats:
    """Aggregate outcome of ingesting a batch of sources."""

    succeeded: int = 0
    failed: int = 0
    chunks_created: int = 0
    elapsed_seconds: float = 0.0
    document_sources: dict[str, str] = field(default_factory=dict[str, str])


@dataclass
class QueryStats:
    """Aggregate latency across a batch of evaluated queries."""

    latencies_ms: list[float] = field(default_factory=list[float])

    @property
    def average_ms(self) -> float:
        """Mean retrieval latency across all evaluated queries.

        Returns:
            Average latency in milliseconds, or 0.0 if no queries ran.
        """
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        """Slowest observed retrieval latency.

        Returns:
            Maximum latency in milliseconds, or 0.0 if no queries ran.
        """
        return max(self.latencies_ms) if self.latencies_ms else 0.0


def _create_langfuse_client() -> Langfuse:
    """Create a Langfuse client, loading ``config/.env`` since it isn't exported by the shell.

    Returns:
        A ``Langfuse`` client with tracing enabled only if credentials are configured.
    """
    load_dotenv("../config/.env")
    has_keys = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    return Langfuse(tracing_enabled=has_keys)


def _load_queries(queries_file: Path | None, ad_hoc: list[str] | None) -> list[QueryCase]:
    """Collect queries to evaluate from a JSON file and/or ad-hoc CLI strings.

    Args:
        queries_file: Optional path to a JSON file containing either a list of strings or a
            list of ``{"text": ..., "category": ...}`` objects.
        ad_hoc: Optional list of raw query strings passed via repeated ``--query`` flags.

    Returns:
        Combined list of ``QueryCase`` entries, file-defined queries first.
    """
    cases: list[QueryCase] = []

    if queries_file is not None:
        raw = json.loads(queries_file.read_text(encoding="utf-8"))
        for item in raw:
            if isinstance(item, str):
                cases.append(QueryCase(text=item))
            else:
                cases.append(QueryCase(text=item["text"], category=item.get("category", "custom")))

    if ad_hoc:
        cases.extend(QueryCase(text=text) for text in ad_hoc)

    return cases


async def _ingest_sources(pipeline: IngestionPipeline, sources: Sequence[str]) -> IngestionStats:
    """Ingest every source into the pipeline's vector store, continuing past failures.

    Args:
        pipeline: Configured ingestion pipeline (loaders, chunker, embedder, vector store).
        sources: PDF file paths or HTML URLs to ingest.

    Returns:
        Aggregate ingestion statistics, including a document_id -> source_uri mapping used
        to attribute retrieval results back to their origin document.
    """
    stats = IngestionStats()
    start = time.perf_counter()

    for source in sources:
        try:
            request = IngestRequest(source_uri=source, source_type=detect_source_type(source))
            response = await pipeline.ingest(request)
        except (IngestionError, ValueError) as exc:
            stats.failed += 1
            _logger.error(f"FAILED {source}: {exc}")
        else:
            stats.succeeded += 1
            stats.chunks_created += response.chunks_created
            stats.document_sources[response.document_id] = source
            _logger.info(f"OK {source}: {response.chunks_created} chunks ({response.document_id})")

    stats.elapsed_seconds = time.perf_counter() - start
    _logger.info(
        f"Ingested {stats.succeeded}/{len(sources)} sources "
        f"({stats.chunks_created} chunks) in {stats.elapsed_seconds:.1f}s"
    )
    return stats


def _format_snippet(text: str) -> str:
    """Collapse whitespace and truncate chunk text for compact display.

    Args:
        text: Raw chunk text.

    Returns:
        Single-line snippet capped at ``_SNIPPET_LENGTH`` characters.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_LENGTH:
        return collapsed
    return f"{collapsed[: _SNIPPET_LENGTH - 3]}..."


def _format_query_section(
    case: QueryCase,
    results: list[Chunk],
    document_sources: dict[str, str],
    latency_ms: float,
) -> str:
    """Render a Markdown section showing one query's top-k retrieved chunks.

    Args:
        case: The evaluated query and its category.
        results: Reranked chunks returned by ``RetrievalService.retrieve``.
        document_sources: Mapping of document_id to source_uri, used to attribute each
            result to the document it was ingested from.
        latency_ms: Wall-clock retrieval latency for this query, in milliseconds.

    Returns:
        Formatted Markdown section with a table of ranked results.
    """
    lines = [
        f"### [{case.category}] {case.text}",
        "",
        f"Latency: `{latency_ms:.0f}ms` | Results: `{len(results)}`",
        "",
        "| Rank | Score | Source | Snippet |",
        "|---|---|---|---|",
    ]
    if not results:
        lines.append("| — | — | — | *no results* |")
    for rank, chunk in enumerate(results, start=1):
        score_str = f"{chunk.score:.4f}" if chunk.score is not None else "n/a"
        source = document_sources.get(chunk.document_id, chunk.document_id[:12])
        snippet = _format_snippet(chunk.text).replace("|", "\\|")
        lines.append(f"| {rank} | {score_str} | `{source}` | {snippet} |")
    lines.append("")
    return "\n".join(lines)


async def _run_queries(
    service: RetrievalService,
    embedder: OllamaEmbedder,
    queries: list[QueryCase],
    document_sources: dict[str, str],
    initial_top_k: int,
    final_top_k: int,
) -> tuple[list[str], QueryStats]:
    """Run every query through the retrieval service and collect per-query reports.

    Args:
        service: Configured two-stage (hybrid search + rerank) retrieval service.
        embedder: Embedder used to compute each query's dense vector.
        queries: Queries to evaluate.
        document_sources: Mapping of document_id to source_uri for result attribution.
        initial_top_k: Broad hybrid candidate pool size before reranking.
        final_top_k: Number of final results to request per query.

    Returns:
        Tuple of (Markdown sections, aggregate latency statistics).
    """
    sections: list[str] = []
    stats = QueryStats()

    for idx, case in enumerate(queries, start=1):
        _logger.info(f"[{idx}/{len(queries)}] ({case.category}) {case.text}")
        embedding = await embedder.embed_text(case.text)
        query = Query(text=case.text, embedding=embedding)

        start = time.perf_counter()
        results = await service.retrieve(
            query, initial_top_k=initial_top_k, final_top_k=final_top_k
        )
        latency_ms = (time.perf_counter() - start) * 1000
        stats.latencies_ms.append(latency_ms)

        section = _format_query_section(case, results, document_sources, latency_ms)
        sections.append(section)
        _logger.info(f"\n{section}")

    return sections, stats


async def _run(
    settings: IngestionSettings,
    collection_name: str,
    sources: list[str],
    queries: list[QueryCase],
    output_path: Path | None,
    *,
    reingest: bool,
    reranker_model: str,
    initial_top_k: int,
    final_top_k: int,
) -> int:
    """Ingest sources (if any) and evaluate queries (if any) against one Qdrant collection.

    Args:
        settings: Connection and chunking configuration.
        collection_name: Target Qdrant collection name.
        sources: PDF paths or HTML URLs to ingest; empty to reuse an already-populated
            collection.
        queries: Queries to evaluate; empty to run an ingestion-only smoke test.
        output_path: Optional Markdown file path to write the full report to.
        reingest: Whether to delete the collection before ingesting.
        reranker_model: FastEmbed cross-encoder model identifier.
        initial_top_k: Broad hybrid candidate pool size before reranking.
        final_top_k: Number of final results requested per query.

    Returns:
        0 on success, 1 if ingestion or retrieval raised an unexpected error.
    """
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    sections: list[str] = ["# Large Corpus Retrieval Test", ""]

    try:
        if reingest and await qdrant.collection_exists(collection_name):
            _logger.info(f"Deleting existing collection '{collection_name}' for re-ingestion...")
            await qdrant.delete_collection(collection_name)

        async with httpx.AsyncClient(
            base_url=settings.ollama_url, timeout=settings.embedding_timeout_seconds
        ) as ollama:
            embedder = OllamaEmbedder(
                ollama, model=settings.embedding_model, batch_size=settings.embedding_batch_size
            )
            vector_store = QdrantVectorStore(qdrant, embedder, collection_name)

            ingestion_stats = IngestionStats()
            if sources:
                pipeline = IngestionPipeline(
                    loaders={
                        SourceType.PDF: PyMuPDFLoader(),
                        SourceType.HTML: TrafilaturaHTMLLoader(),
                    },
                    chunker=LangchainTextChunker(settings.chunk_size, settings.chunk_overlap),
                    embedder=embedder,
                    vector_store=vector_store,
                )
                ingestion_stats = await _ingest_sources(pipeline, sources)

            point_count = 0
            if await qdrant.collection_exists(collection_name):
                point_count = (await qdrant.count(collection_name, exact=True)).count

            sections.append(
                f"Collection: `{collection_name}` | Points: `{point_count}` | "
                f"Ingested this run: `{ingestion_stats.succeeded}` sources / "
                f"`{ingestion_stats.chunks_created}` chunks "
                f"(`{ingestion_stats.elapsed_seconds:.1f}s`, "
                f"`{ingestion_stats.failed}` failed)"
            )
            sections.append("")

            query_stats = QueryStats()
            if queries:
                reranker = FastEmbedReranker(model_name=reranker_model)
                service = RetrievalService(
                    embedder=embedder,
                    vector_store=vector_store,
                    reranker=reranker,
                    langfuse_client=_create_langfuse_client(),
                )
                query_sections, query_stats = await _run_queries(
                    service,
                    embedder,
                    queries,
                    ingestion_stats.document_sources,
                    initial_top_k,
                    final_top_k,
                )
                sections.append(
                    f"Queries: `{len(queries)}` | Avg latency: `{query_stats.average_ms:.0f}ms` | "
                    f"Max latency: `{query_stats.max_ms:.0f}ms`"
                )
                sections.append("")
                sections.extend(query_sections)
    except Exception as exc:
        _logger.error(f"Large corpus test failed: {exc}")
        return _EXIT_FAILURE
    finally:
        await qdrant.close()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(sections), encoding="utf-8")
        _logger.info(f"Wrote large corpus test report to {output_path}")

    return _EXIT_OK


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the large corpus ingestion + retrieval test.

    Args:
        argv: Optional argument sequence; defaults to the process's own arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="ingest_e2e",
        description="Ingest many PDFs/HTML pages and evaluate retrieval against them.",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        metavar="SOURCE",
        help="PDF file path(s) or HTML URL(s) to ingest. Omit to reuse an existing collection.",
    )
    parser.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help="Target Qdrant collection name.",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Delete the collection before ingesting the given sources.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="JSON file with a list of query strings or {text, category} objects.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="query",
        default=None,
        help="Ad-hoc query string to evaluate; repeat for multiple queries.",
    )
    parser.add_argument(
        "--reranker-model",
        default=_DEFAULT_RERANKER_MODEL,
        help="FastEmbed cross-encoder model identifier.",
    )
    parser.add_argument(
        "--initial-top-k",
        type=int,
        default=_DEFAULT_INITIAL_TOP_K,
        help="Number of hybrid candidates retrieved before reranking.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=_DEFAULT_FINAL_TOP_K,
        help="Number of top results requested per query.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown file path to write the ingestion + retrieval report to.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the large corpus ingestion + retrieval smoke test.

    Args:
        argv: Optional command line arguments.

    Returns:
        0 on success, 1 on failure.
    """
    setup_logging()
    args = _parse_args(argv)
    settings = IngestionSettings()
    queries = _load_queries(args.queries_file, args.query)

    if not args.sources and not queries:
        _logger.error("Nothing to do: pass SOURCE(s) to ingest and/or --query/--queries-file.")
        return _EXIT_FAILURE

    return asyncio.run(
        _run(
            settings,
            args.collection,
            args.sources,
            queries,
            args.output,
            reingest=args.reingest,
            reranker_model=args.reranker_model,
            initial_top_k=args.initial_top_k,
            final_top_k=args.final_top_k,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
