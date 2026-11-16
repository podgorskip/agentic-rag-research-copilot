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
    # Then
    assert isinstance(mock_loader, DocumentLoaderProtocol)
    assert isinstance(mock_chunker, ChunkerProtocol)
    assert isinstance(mock_embedder, EmbedderProtocol)
    assert isinstance(in_memory_store, VectorStoreProtocol)


async def test_loader_synthesises_stable_documents(mock_loader: MockDocumentLoader) -> None:
    # When
    first = await mock_loader.load("a.pdf")
    again = await mock_loader.load("a.pdf")
    other = await mock_loader.load("b.pdf")

    # Then
    assert first == again
    assert first.id != other.id
    assert first.source_type is SourceType.PDF
    assert mock_loader.loaded_uris == ["a.pdf", "a.pdf", "b.pdf"]


async def test_loader_returns_configured_document() -> None:
    # Given
    document = make_document()
    loader = MockDocumentLoader({"known": document})

    # When
    loaded = await loader.load("known")

    # Then
    assert loaded is document


async def test_loader_raises_configured_failure() -> None:
    # Given
    loader = MockDocumentLoader(failures={"bad.pdf": FileNotFoundError("missing")})

    # When / Then
    with pytest.raises(FileNotFoundError, match="missing"):
        await loader.load("bad.pdf")


def test_chunker_windows_text_and_inherits_metadata() -> None:
    # Given
    chunker = MockChunker(chunk_size=4)
    document = make_document(author="A")

    # When
    chunks = chunker.chunk(document)

    # Then
    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]
    assert {chunk.document_id for chunk in chunks} == {"doc"}
    assert len({chunk.id for chunk in chunks}) == 3
    assert chunks[2].metadata == {"author": "A", "chunk_index": 2}


def test_chunker_raises_configured_failure() -> None:
    # Given
    chunker = MockChunker(fail_with=ValueError("boom"))

    # When / Then
    with pytest.raises(ValueError, match="boom"):
        chunker.chunk(make_document())


async def test_embedder_is_deterministic_and_normalised() -> None:
    # Given
    embedder = MockEmbedder(dimension=16)

    # When
    first, same, other = await embedder.embed_batch(["x", "x", "y"])

    # Then
    assert first == same
    assert first != other
    assert len(first) == 16
    assert math.hypot(*first) == pytest.approx(1.0)
    assert embedder.batches == [["x", "x", "y"]]


async def test_embed_text_matches_embed_batch(mock_embedder: MockEmbedder) -> None:
    # When
    text_result = await mock_embedder.embed_text("x")
    batch_result = await mock_embedder.embed_batch(["x"])

    # Then
    assert text_result == batch_result[0]


async def test_embedder_raises_configured_failure() -> None:
    # Given
    embedder = MockEmbedder(fail_with=ConnectionError())

    # When / Then
    with pytest.raises(ConnectionError):
        await embedder.embed_text("x")


async def test_store_keeps_latest_version_of_chunk(in_memory_store: InMemoryVectorStore) -> None:
    # When
    await in_memory_store.upsert([Chunk(id="c", document_id="d", text="old", embedding=[1.0])])
    await in_memory_store.upsert([Chunk(id="c", document_id="d", text="new", embedding=[1.0])])
    results = await in_memory_store.hybrid_search(Query(text="q"), top_k=5)

    # Then
    assert [chunk.text for chunk in results] == ["new"]


async def test_store_rejects_chunk_without_embedding(in_memory_store: InMemoryVectorStore) -> None:
    # When / Then
    with pytest.raises(ValueError, match="no embedding"):
        await in_memory_store.upsert([Chunk(id="c", document_id="d", text="t")])


async def test_store_raises_configured_failure() -> None:
    # Given
    store = InMemoryVectorStore(fail_with=ConnectionError())

    # When / Then
    with pytest.raises(ConnectionError):
        await store.upsert([])
