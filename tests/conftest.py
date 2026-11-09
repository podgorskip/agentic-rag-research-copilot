"""Shared pytest fixtures."""

import pytest

from tests.mocks.ingestion import (
    InMemoryVectorStore,
    MockChunker,
    MockDocumentLoader,
    MockEmbedder,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_loader() -> MockDocumentLoader:
    return MockDocumentLoader()


@pytest.fixture
def mock_chunker() -> MockChunker:
    return MockChunker()


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def in_memory_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()
