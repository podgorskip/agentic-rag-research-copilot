"""Shared fixtures for agent grading and rewriting integration tests.

Uses a real local Ollama instance (LLM + embeddings), FastEmbedReranker, and an
in-memory Qdrant instance seeded from the Milestone 03 validation corpus
(``arXiv:2005.11401`` cached at ``.cache/validation/2005.11401.pdf``).
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import httpx
import pytest
from langfuse import Langfuse
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.embedding import OllamaEmbedder
from agentic_rag_copilot.adapters.ingestion import LangchainTextChunker, PyMuPDFLoader
from agentic_rag_copilot.adapters.llm import OllamaLLM
from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker
from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.agent import AgentSettings
from agentic_rag_copilot.domain.models import Chunk
from agentic_rag_copilot.ingestion import IngestionSettings
from agentic_rag_copilot.services import RetrievalService

AGENT_SETTINGS: Final[AgentSettings] = AgentSettings()
_INGESTION_SETTINGS: Final[IngestionSettings] = IngestionSettings()
_VALIDATION_PDF_PATH: Final[Path] = Path(".cache/validation/2005.11401.pdf")
_INTEGRATION_COLLECTION: Final[str] = "integration_lewis_rag_2020"


def ollama_is_reachable() -> bool:
    """Check whether the configured Ollama server responds on ``/api/tags``.

    Returns:
        ``True`` if Ollama responds with a 2xx status, ``False`` otherwise.
    """
    try:
        httpx.get(f"{AGENT_SETTINGS.ollama_url}/api/tags", timeout=2.0).raise_for_status()
    except httpx.HTTPError:
        return False
    return True


async def _load_and_embed_validation_chunks() -> list[Chunk]:
    """Parse, chunk, and embed the cached Milestone 03 validation PDF once.

    Returns:
        List of ``Chunk`` instances with dense embeddings populated.

    Raises:
        FileNotFoundError: If the cached validation PDF does not exist.
    """
    if not _VALIDATION_PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Validation PDF not found at {_VALIDATION_PDF_PATH}; "
            "run 'just validate-retrieval' first to cache arXiv:2005.11401."
        )

    document = await PyMuPDFLoader().load(str(_VALIDATION_PDF_PATH))
    chunks = LangchainTextChunker(
        chunk_size=_INGESTION_SETTINGS.chunk_size,
        chunk_overlap=_INGESTION_SETTINGS.chunk_overlap,
    ).chunk(document)

    async with httpx.AsyncClient(
        base_url=_INGESTION_SETTINGS.ollama_url,
        timeout=_INGESTION_SETTINGS.embedding_timeout_seconds,
    ) as client:
        embedder = OllamaEmbedder(
            client,
            model=_INGESTION_SETTINGS.embedding_model,
            batch_size=_INGESTION_SETTINGS.embedding_batch_size,
        )
        embeddings = await embedder.embed_batch([chunk.text for chunk in chunks])

    return [
        chunk.model_copy(update={"embedding": embedding})
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


@pytest.fixture(scope="session")
def reranker() -> FastEmbedReranker:
    """Reuse a single FastEmbedReranker instance across the test session.

    Returns:
        Configured ``FastEmbedReranker`` instance.
    """
    return FastEmbedReranker()


@pytest.fixture(scope="session")
def embedded_validation_chunks() -> list[Chunk]:
    """Provide pre-embedded chunks from ``arXiv:2005.11401`` for agent integration tests.

    Returns:
        List of embedded ``Chunk`` instances.
    """
    return asyncio.run(_load_and_embed_validation_chunks())


@pytest.fixture
async def ollama_llm() -> AsyncIterator[OllamaLLM]:
    """Yield a real ``OllamaLLM`` adapter bound to the configured local Ollama server.

    Yields:
        Configured ``OllamaLLM`` instance.
    """
    async with httpx.AsyncClient(
        base_url=AGENT_SETTINGS.ollama_url,
        timeout=AGENT_SETTINGS.llm_timeout_seconds,
    ) as client:
        yield OllamaLLM(client, model=AGENT_SETTINGS.llm_model)


@pytest.fixture
async def retrieval_service(
    embedded_validation_chunks: list[Chunk],
    reranker: FastEmbedReranker,
) -> AsyncIterator[RetrievalService]:
    """Yield a real ``RetrievalService`` backed by in-memory Qdrant and the validation corpus.

    Args:
        embedded_validation_chunks: Pre-embedded validation corpus chunks.
        reranker: Session-scoped cross-encoder reranker.

    Yields:
        Configured ``RetrievalService`` instance.
    """
    qdrant = AsyncQdrantClient(":memory:")
    try:
        async with httpx.AsyncClient(
            base_url=_INGESTION_SETTINGS.ollama_url,
            timeout=_INGESTION_SETTINGS.embedding_timeout_seconds,
        ) as ollama_client:
            embedder = OllamaEmbedder(
                ollama_client,
                model=_INGESTION_SETTINGS.embedding_model,
                batch_size=_INGESTION_SETTINGS.embedding_batch_size,
            )
            store = QdrantVectorStore(
                qdrant,
                embedder=embedder,
                collection_name=_INTEGRATION_COLLECTION,
            )
            await store.upsert(embedded_validation_chunks)
            yield RetrievalService(
                embedder=embedder,
                vector_store=store,
                reranker=reranker,
                langfuse_client=Langfuse(tracing_enabled=False),
            )
    finally:
        await qdrant.close()
