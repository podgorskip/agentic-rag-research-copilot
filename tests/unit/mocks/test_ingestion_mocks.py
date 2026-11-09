import math

import pytest

from agentic_rag_copilot.domain.models import Chunk, Document, Query, SourceType
from agentic_rag_copilot.domain.protocols import (
    ChunkerProtocol,
    DocumentLoaderProtocol,
    EmbedderProtocol,
    VectorStoreProtocol,
)
from tests.mocks.ingestion import (
    InMemoryVectorStore,
    MockChunker,
    MockDocumentLoader,
    MockEmbedder,
)


def make_document(content: str = "abcdefghij", **metadata: object) -> Document:
    return Document(
        id="doc",
        source_type=SourceType.HTML,
        source_uri="https://example.test",
        raw_content=content,
        metadata=metadata,
    )


def test_mocks_satisfy_protocols(
    mock_loader: MockDocumentLoader,
    mock_chunker: MockChunker,
    mock_embedder: MockEmbedder,
    in_memory_store: InMemoryVectorStore,
) -> None:
    assert isinstance(mock_loader, DocumentLoaderProtocol)
    assert isinstance(mock_chunker, ChunkerProtocol)
    assert isinstance(mock_embedder, EmbedderProtocol)
    assert isinstance(in_memory_store, VectorStoreProtocol)


async def test_loader_synthesises_stable_documents(mock_loader: MockDocumentLoader) -> None:
    first = await mock_loader.load("a.pdf")
    again = await mock_loader.load("a.pdf")
    other = await mock_loader.load("b.pdf")

    assert first == again
    assert first.id != other.id
    assert first.source_type is SourceType.PDF
    assert mock_loader.loaded_uris == ["a.pdf", "a.pdf", "b.pdf"]


async def test_loader_returns_configured_document() -> None:
    document = make_document()

    assert await MockDocumentLoader({"known": document}).load("known") is document


async def test_loader_raises_configured_failure() -> None:
    loader = MockDocumentLoader(failures={"bad.pdf": FileNotFoundError("missing")})

    with pytest.raises(FileNotFoundError, match="missing"):
        await loader.load("bad.pdf")


def test_chunker_windows_text_and_inherits_metadata() -> None:
    chunks = MockChunker(chunk_size=4).chunk(make_document(author="A"))

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]
    assert {chunk.document_id for chunk in chunks} == {"doc"}
    assert len({chunk.id for chunk in chunks}) == 3
    assert chunks[2].metadata == {"author": "A", "chunk_index": 2}


def test_chunker_raises_configured_failure() -> None:
    with pytest.raises(ValueError, match="boom"):
        MockChunker(fail_with=ValueError("boom")).chunk(make_document())


async def test_embedder_is_deterministic_and_normalised() -> None:
    embedder = MockEmbedder(dimension=16)

    first, same, other = await embedder.embed_batch(["x", "x", "y"])

    assert first == same
    assert first != other
    assert len(first) == 16
    assert math.hypot(*first) == pytest.approx(1.0)
    assert embedder.batches == [["x", "x", "y"]]


async def test_embed_text_matches_embed_batch(mock_embedder: MockEmbedder) -> None:
    assert await mock_embedder.embed_text("x") == (await mock_embedder.embed_batch(["x"]))[0]


async def test_embedder_raises_configured_failure() -> None:
    with pytest.raises(ConnectionError):
        await MockEmbedder(fail_with=ConnectionError()).embed_text("x")


async def test_store_keeps_latest_version_of_chunk(in_memory_store: InMemoryVectorStore) -> None:
    await in_memory_store.upsert([Chunk(id="c", document_id="d", text="old", embedding=[1.0])])
    await in_memory_store.upsert([Chunk(id="c", document_id="d", text="new", embedding=[1.0])])

    results = await in_memory_store.hybrid_search(Query(text="q"), top_k=5)

    assert [chunk.text for chunk in results] == ["new"]


async def test_store_rejects_chunk_without_embedding(in_memory_store: InMemoryVectorStore) -> None:
    with pytest.raises(ValueError, match="no embedding"):
        await in_memory_store.upsert([Chunk(id="c", document_id="d", text="t")])


async def test_store_raises_configured_failure() -> None:
    with pytest.raises(ConnectionError):
        await InMemoryVectorStore(fail_with=ConnectionError()).upsert([])
