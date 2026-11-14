"""Standalone retrieval quality validation against Lewis et al. (2020) RAG paper."""

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import urlparse

import httpx
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
from agentic_rag_copilot.ingestion import IngestionPipeline, IngestionSettings, detect_source_type
from agentic_rag_copilot.logging import get_logger, setup_logging
from agentic_rag_copilot.services import RetrievalService

logger = get_logger(__name__)

_DEFAULT_SOURCE_URI: Final[str] = "https://arxiv.org/pdf/2005.11401"
_DEFAULT_CACHE_DIR: Final[Path] = Path(".cache/validation")
_DEFAULT_COLLECTION: Final[str] = "validation_lewis_rag_2020"
_DEFAULT_RERANKER_MODEL: Final[str] = "jinaai/jina-reranker-v2-base-multilingual"
_PDF_DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 60.0
_DENSE_VECTOR_NAME: Final[str] = "dense"
_INITIAL_TOP_K: Final[int] = 30
_FINAL_TOP_K: Final[int] = 6
_SNIPPET_LENGTH: Final[int] = 80
_EXIT_OK: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1


@dataclass(frozen=True)
class ValidationQuery:
    """A labeled validation query belonging to a retrieval evaluation category."""

    category: str
    text: str


_DEFAULT_QUERIES: Final[tuple[ValidationQuery, ...]] = (
    ValidationQuery(
        category="exact_term",
        text="RAG-Sequence RAG-Token marginalization equation",
    ),
    ValidationQuery(
        category="exact_term",
        text="DPR Dense Passage Retriever BERT-base 100-word Wikipedia",
    ),
    ValidationQuery(
        category="exact_term",
        text="MIPS Maximum Inner Product Search FAISS HNSW",
    ),
    ValidationQuery(
        category="exact_term",
        text="MSMARCO Jeopardy Question Generation B-1 Q-L ROUGE-L",
    ),
    ValidationQuery(
        category="conceptual",
        text="What is the difference between parametric memory and non-parametric memory in RAG?",
    ),
    ValidationQuery(
        category="conceptual",
        text="How does RAG-Token differ from RAG-Sequence when generating a target sequence?",
    ),
    ValidationQuery(
        category="conceptual",
        text="Why does thorough decoding or fast decoding matter for RAG-Sequence?",
    ),
    ValidationQuery(
        category="conceptual",
        text="How does RAG reduce factual hallucinations compared to BART alone?",
    ),
    ValidationQuery(
        category="multi_hop",
        text=(
            "Which retriever architecture and Wikipedia index size initialize "
            "non-parametric memory?"
        ),
    ),
    ValidationQuery(
        category="multi_hop",
        text="How does RAG perform on FEVER fact verification without supervised retrieval?",
    ),
    ValidationQuery(
        category="multi_hop",
        text="What happens during RAG training when the document encoder is frozen vs fine-tuned?",
    ),
    ValidationQuery(
        category="multi_hop",
        text="How does index hot-swapping update world knowledge without retraining the generator?",
    ),
)


def _create_langfuse_client() -> Langfuse:
    """Create a Langfuse client, disabling tracing if credentials are unset.

    Returns:
        Configured ``Langfuse`` instance.
    """
    has_keys = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    return Langfuse(tracing_enabled=has_keys)


async def _resolve_source_uri(source_uri: str, cache_dir: Path = _DEFAULT_CACHE_DIR) -> str:
    """Resolve a PDF URL to a local cached file path, downloading only if missing.

    Args:
        source_uri: Local file path or HTTP(S) URL.
        cache_dir: Directory where downloaded PDFs are cached locally.

    Returns:
        Local file path for remote PDFs, or the original ``source_uri`` otherwise.
    """
    is_remote = source_uri.startswith(("http://", "https://"))
    if not is_remote or detect_source_type(source_uri) is not SourceType.PDF:
        return source_uri

    raw_name = Path(urlparse(source_uri).path).name or "validation_paper.pdf"
    filename = raw_name if raw_name.lower().endswith(".pdf") else f"{raw_name}.pdf"
    target_path = cache_dir / filename

    if target_path.is_file() and target_path.stat().st_size > 0:
        logger.info(f"Using cached validation PDF at '{target_path}'.")
        return str(target_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading validation PDF from '{source_uri}' to '{target_path}'...")
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=_PDF_DOWNLOAD_TIMEOUT_SECONDS,
    ) as client:
        response = await client.get(source_uri)
        response.raise_for_status()
        target_path.write_bytes(response.content)

    return str(target_path)


async def _ensure_validation_corpus(
    qdrant: AsyncQdrantClient,
    embedder: OllamaEmbedder,
    vector_store: QdrantVectorStore,
    settings: IngestionSettings,
    collection_name: str,
    source_uri: str,
    *,
    reingest: bool,
) -> None:
    """Ingest the validation paper if the target collection is missing or reingest is requested.

    Args:
        qdrant: Async Qdrant client.
        embedder: Configured Ollama embedder.
        vector_store: Target Qdrant vector store bound to ``collection_name``.
        settings: Chunking configuration.
        collection_name: Name of the Qdrant collection used for validation.
        source_uri: PDF path or URL to ingest.
        reingest: Whether to delete and rebuild an existing collection.
    """
    exists = await qdrant.collection_exists(collection_name)
    if exists and reingest:
        logger.info(f"Deleting existing collection '{collection_name}' for re-ingestion...")
        await qdrant.delete_collection(collection_name)
        exists = False

    if exists:
        point_count = (await qdrant.count(collection_name, exact=True)).count
        if point_count > 0:
            logger.info(f"Reusing collection '{collection_name}' ({point_count} chunks).")
            return

    resolved_uri = await _resolve_source_uri(source_uri)
    logger.info(f"Ingesting validation corpus '{resolved_uri}' into '{collection_name}'...")
    pipeline = IngestionPipeline(
        loaders={
            SourceType.PDF: PyMuPDFLoader(),
            SourceType.HTML: TrafilaturaHTMLLoader(),
        },
        chunker=LangchainTextChunker(settings.chunk_size, settings.chunk_overlap),
        embedder=embedder,
        vector_store=vector_store,
    )
    request = IngestRequest(source_uri=resolved_uri, source_type=detect_source_type(resolved_uri))
    response = await pipeline.ingest(request)
    logger.info(
        f"Ingested '{source_uri}': {response.chunks_created} chunks (doc={response.document_id})"
    )


async def _dense_only_search(
    client: AsyncQdrantClient,
    collection_name: str,
    embedding: list[float],
    top_k: int = _FINAL_TOP_K,
) -> list[Chunk]:
    """Query only the dense vector index in Qdrant for baseline comparison.

    Args:
        client: Async Qdrant client.
        collection_name: Target Qdrant collection name.
        embedding: Precomputed dense query vector.
        top_k: Number of top chunks to return.

    Returns:
        Top-k chunks ordered by dense cosine similarity.
    """
    response = await client.query_points(
        collection_name=collection_name,
        query=embedding,
        using=_DENSE_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    )
    chunks: list[Chunk] = []
    for point in response.points:
        payload = point.payload or {}
        raw_meta = payload.get("metadata")
        metadata: dict[str, Any] = (
            dict(cast("dict[str, Any]", raw_meta)) if isinstance(raw_meta, dict) else {}
        )
        chunks.append(
            Chunk(
                id=str(payload.get("chunk_id", point.id)),
                document_id=str(payload.get("document_id", "")),
                text=str(payload.get("text", "")),
                metadata=metadata,
                score=float(point.score),
            )
        )
    return chunks


def _format_snippet(text: str) -> str:
    """Collapse whitespace and truncate chunk text for compact side-by-side display.

    Args:
        text: Raw chunk text.

    Returns:
        Single-line snippet capped at ``_SNIPPET_LENGTH`` characters.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_LENGTH:
        return collapsed
    return f"{collapsed[: _SNIPPET_LENGTH - 3]}..."


def _format_comparison_markdown(
    vq: ValidationQuery,
    dense_chunks: list[Chunk],
    hybrid_chunks: list[Chunk],
    reranked_chunks: list[Chunk],
    final_top_k: int = _FINAL_TOP_K,
) -> str:
    """Render a side-by-side top-k Markdown table for one validation query.

    Args:
        vq: Validation query metadata.
        dense_chunks: Top-k dense-only results.
        hybrid_chunks: Top-k hybrid (dense + BM25) results.
        reranked_chunks: Top-k hybrid + cross-encoder reranked results.
        final_top_k: Number of ranked rows to render.

    Returns:
        Formatted Markdown section comparing the three rankings.
    """
    lines = [
        f"### [{vq.category}] {vq.text}",
        "",
        "| Rank | Dense-only | Hybrid (Dense + BM25) | Hybrid + Reranked |",
        "|---|---|---|---|",
    ]
    for rank in range(final_top_k):
        row_cells: list[str] = []
        for group in (dense_chunks, hybrid_chunks, reranked_chunks):
            if rank < len(group):
                chunk = group[rank]
                score_str = f"{chunk.score:.4f}" if chunk.score is not None else "n/a"
                snippet = _format_snippet(chunk.text).replace("|", "\\|")
                row_cells.append(f"`{chunk.id[:8]}` ({score_str}): {snippet}")
            else:
                row_cells.append("—")
        lines.append(f"| {rank + 1} | {row_cells[0]} | {row_cells[1]} | {row_cells[2]} |")
    lines.append("")
    return "\n".join(lines)


def _load_queries(queries_file: Path | None) -> list[ValidationQuery]:
    """Load validation queries from a JSON file or return the built-in defaults.

    Args:
        queries_file: Optional path to JSON list of ``{"category": ..., "text": ...}`` objects.

    Returns:
        List of ``ValidationQuery`` entries to evaluate.
    """
    if queries_file is None:
        return list(_DEFAULT_QUERIES)

    raw = json.loads(queries_file.read_text(encoding="utf-8"))
    return [ValidationQuery(category=item["category"], text=item["text"]) for item in raw]


async def _run_validation(
    settings: IngestionSettings,
    collection_name: str,
    source_uri: str,
    queries: list[ValidationQuery],
    output_path: Path | None,
    *,
    reingest: bool,
    reranker_model: str = _DEFAULT_RERANKER_MODEL,
    initial_top_k: int = _INITIAL_TOP_K,
    final_top_k: int = _FINAL_TOP_K,
) -> int:
    """Ingest the validation corpus if needed and log side-by-side retrieval comparisons.

    Args:
        settings: Connection and model configuration.
        collection_name: Target Qdrant collection name.
        source_uri: Source PDF/HTML URI to ingest if the collection is empty/missing.
        queries: Queries to evaluate.
        output_path: Optional file path to write the Markdown comparison report.
        reingest: Whether to rebuild the validation collection before running queries.
        reranker_model: FastEmbed cross-encoder model name.
        initial_top_k: Broad hybrid candidate pool size before reranking.
        final_top_k: Number of final results to display per retrieval mode.

    Returns:
        0 on success, 1 if ingestion or retrieval fails.
    """
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    sections: list[str] = [
        "# Retrieval Quality Validation Output",
        "",
        (
            f"Source: `{source_uri}` | Collection: `{collection_name}` | "
            f"Reranker: `{reranker_model}` | `initial_top_k={initial_top_k}` | "
            f"`final_top_k={final_top_k}`"
        ),
        "",
    ]

    try:
        async with httpx.AsyncClient(
            base_url=settings.ollama_url, timeout=settings.embedding_timeout_seconds
        ) as ollama:
            embedder = OllamaEmbedder(
                ollama,
                model=settings.embedding_model,
                batch_size=settings.embedding_batch_size,
            )
            vector_store = QdrantVectorStore(qdrant, embedder, collection_name)
            await _ensure_validation_corpus(
                qdrant,
                embedder,
                vector_store,
                settings,
                collection_name,
                source_uri,
                reingest=reingest,
            )

            reranker = FastEmbedReranker(model_name=reranker_model)
            service = RetrievalService(
                embedder=embedder,
                vector_store=vector_store,
                reranker=reranker,
                langfuse_client=_create_langfuse_client(),
            )

            for idx, vq in enumerate(queries, start=1):
                logger.info(f"[{idx}/{len(queries)}] ({vq.category}) {vq.text}")
                embedding = await embedder.embed_text(vq.text)
                query = Query(text=vq.text, embedding=embedding)

                dense_results = await _dense_only_search(
                    qdrant, collection_name, embedding, top_k=final_top_k
                )
                hybrid_results = await vector_store.hybrid_search(query, top_k=final_top_k)
                reranked_results = await service.retrieve(
                    query, initial_top_k=initial_top_k, final_top_k=final_top_k
                )

                section = _format_comparison_markdown(
                    vq,
                    dense_results,
                    hybrid_results,
                    reranked_results,
                    final_top_k=final_top_k,
                )
                sections.append(section)
                logger.info(f"\n{section}")
    except Exception as exc:
        logger.error(f"Validation failed: {exc}")
        return _EXIT_FAILURE
    finally:
        await qdrant.close()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(sections), encoding="utf-8")
        logger.info(f"Wrote validation comparison report to {output_path}")

    return _EXIT_OK


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the retrieval validation script.

    Args:
        argv: Optional argument sequence.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="validate_retrieval",
        description="Standalone comparison of Dense vs. Hybrid vs. Hybrid + Reranked results.",
    )
    parser.add_argument(
        "--source",
        default=_DEFAULT_SOURCE_URI,
        help="PDF path or URL to ingest when building the validation collection.",
    )
    parser.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help="Dedicated Qdrant collection name for validation.",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Delete and re-ingest the validation collection before running queries.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="Optional JSON file containing custom validation queries.",
    )
    parser.add_argument(
        "--reranker-model",
        default=_DEFAULT_RERANKER_MODEL,
        help="FastEmbed cross-encoder model identifier.",
    )
    parser.add_argument(
        "--initial-top-k",
        type=int,
        default=_INITIAL_TOP_K,
        help="Number of hybrid candidates retrieved before reranking.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=_FINAL_TOP_K,
        help="Number of top results displayed per retrieval mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown file path to save the side-by-side comparison tables.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run standalone retrieval quality validation.

    Args:
        argv: Optional command line arguments.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    setup_logging()
    args = _parse_args(argv)
    settings = IngestionSettings()
    queries = _load_queries(args.queries_file)
    return asyncio.run(
        _run_validation(
            settings,
            args.collection,
            args.source,
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
