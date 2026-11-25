"""Shared fixtures and helpers for deterministic integration tests."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import pytest
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker

SAMPLE_PAPER_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "e2e" / "data" / "paper-01.pdf"
)


@pytest.fixture(scope="session")
def shared_reranker() -> FastEmbedReranker:
    """Reuse a single ``FastEmbedReranker`` instance across the integration test session.

    Returns:
        Configured ``FastEmbedReranker``.
    """
    return FastEmbedReranker()


@pytest.fixture
async def in_memory_qdrant() -> AsyncIterator[AsyncQdrantClient]:
    """Provide an isolated in-memory Qdrant client per test.

    Yields:
        ``AsyncQdrantClient`` backed by in-memory storage.
    """
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


async def stored_chunk_ids(qdrant: AsyncQdrantClient, collection: str) -> set[str]:
    """Collect all ``chunk_id`` payload values currently stored in ``collection``.

    Args:
        qdrant: Qdrant client to scroll.
        collection: Target collection name.

    Returns:
        Set of stored chunk IDs.
    """
    points, _ = await qdrant.scroll(collection, limit=500, with_payload=True)
    return {
        str(point.payload["chunk_id"])
        for point in points
        if point.payload is not None and "chunk_id" in point.payload
    }
