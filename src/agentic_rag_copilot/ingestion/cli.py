"""Command line entry point for the end-to-end ingestion flow."""

import argparse
import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Final

import httpx
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.embedding import OllamaEmbedder
from agentic_rag_copilot.adapters.ingestion import (
    LangchainTextChunker,
    PyMuPDFLoader,
    TrafilaturaHTMLLoader,
)
from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.domain.models import IngestRequest, SourceType
from agentic_rag_copilot.logging import get_logger, setup_logging

from ._pipeline import IngestionError, IngestionPipeline, detect_source_type
from ._settings import IngestionSettings

logger = get_logger(__name__)

_EXIT_OK: Final[int] = 0
_EXIT_INGESTION_FAILED: Final[int] = 1
_EXIT_INVALID_CONFIG: Final[int] = 2


@asynccontextmanager
async def build_pipeline(settings: IngestionSettings) -> AsyncGenerator[IngestionPipeline]:
    """Wire the concrete adapters together and release their connections on exit.

    Args:
        settings: Connection details and tuning parameters.

    Yields:
        A pipeline ready to ingest sources.

    Raises:
        ValueError: If the chunking parameters are inconsistent.
    """
    chunker = LangchainTextChunker(settings.chunk_size, settings.chunk_overlap)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        async with httpx.AsyncClient(
            base_url=settings.ollama_url, timeout=settings.embedding_timeout_seconds
        ) as ollama:
            yield IngestionPipeline(
                loaders={
                    SourceType.PDF: PyMuPDFLoader(),
                    SourceType.HTML: TrafilaturaHTMLLoader(),
                },
                chunker=chunker,
                embedder=OllamaEmbedder(
                    ollama,
                    model=settings.embedding_model,
                    batch_size=settings.embedding_batch_size,
                ),
                vector_store=QdrantVectorStore(qdrant, settings.qdrant_collection),
            )
    finally:
        await qdrant.close()


async def _ingest_sources(pipeline: IngestionPipeline, sources: Sequence[str]) -> int:
    """Ingest every source, continuing past failures so one bad file cannot abort a batch.

    Args:
        pipeline: The ingestion pipeline with configured adapters.
        sources: The source URIs to ingest.

    Returns:
        0 if all sources were ingested successfully, 1 if any failed.
    """
    failed = 0
    for source in sources:
        try:
            request = IngestRequest(source_uri=source, source_type=detect_source_type(source))
            response = await pipeline.ingest(request)
        except (IngestionError, ValueError) as exc:
            failed += 1
            logger.error(f"FAILED {source}: {exc}")
        else:
            logger.info(f"OK {source}: {response.chunks_created} chunks ({response.document_id})")

    logger.info(f"Ingested {len(sources) - failed} of {len(sources)} sources")
    return _EXIT_INGESTION_FAILED if failed else _EXIT_OK


async def _run(settings: IngestionSettings, sources: Sequence[str]) -> int:
    """Build the ingestion pipeline and ingest sources.

    Args:
        settings: Connection details and tuning parameters.
        sources: The source URIs to ingest.

    Returns:
        0 if all sources were ingested successfully, 1 if any failed.
    """
    async with build_pipeline(settings) as pipeline:
        return await _ingest_sources(pipeline, sources)


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Command line arguments; defaults to those of the current process.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Load, chunk, embed and upsert documents into Qdrant.",
    )
    parser.add_argument("sources", nargs="+", metavar="SOURCE", help="PDF file path or HTML URL")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and ingest the requested sources.

    Args:
        argv: Command line arguments; defaults to those of the current process.

    Returns:
        0 on success, 1 if any source failed, 2 if the configuration is invalid.
    """
    setup_logging()

    args = _parse_arguments(argv)

    try:
        settings = IngestionSettings()
        return asyncio.run(_run(settings, args.sources))
    except ValueError as exc:
        logger.error(f"Invalid configuration: {exc}")
        return _EXIT_INVALID_CONFIG
