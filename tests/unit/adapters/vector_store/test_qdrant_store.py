import uuid
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient, models

from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.domain.models import Chunk, Query
from agentic_rag_copilot.domain.protocols import EmbedderProtocol, VectorStoreProtocol
from tests.mocks.ingestion import MockEmbedder

COLLECTION = "test_chunks"
VECTOR_SIZE = 3
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
BM25_MODEL = "Qdrant/bm25"
POINT_ID_NAMESPACE = uuid.UUID("6f1c2a8e-3b4d-4e5f-9a7b-1c2d3e4f5a6b")


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    qdrant = AsyncQdrantClient(":memory:")
    yield qdrant
    await qdrant.close()


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder(dimension=VECTOR_SIZE)


@pytest.fixture
def store(client: AsyncQdrantClient, embedder: MockEmbedder) -> QdrantVectorStore:
    return QdrantVectorStore(client, embedder=embedder, collection_name=COLLECTION)


def make_chunk(chunk_id: str, embedding: list[float] | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc_1",
        text=f"text for {chunk_id}",
        metadata={"chunk_index": 0, "page": 3},
        embedding=[0.6, 0.0, 0.8] if embedding is None else embedding,
    )


def _mock_embedder(
    *,
    embed_text_return: list[float] | None = None,
    embed_batch_return: list[list[float]] | None = None,
    embed_text_side_effect: Exception | None = None,
    embed_batch_side_effect: Exception | None = None,
) -> AsyncMock:
    mock = AsyncMock(spec=EmbedderProtocol)
    mock.embed_text = AsyncMock(
        return_value=embed_text_return or [0.1, 0.2, 0.3],
        side_effect=embed_text_side_effect,
    )
    mock.embed_batch = AsyncMock(
        return_value=embed_batch_return if embed_batch_return is not None else [[0.1, 0.2, 0.3]],
        side_effect=embed_batch_side_effect,
    )
    return mock


def _mock_qdrant_client() -> AsyncMock:
    mock = AsyncMock(spec=AsyncQdrantClient)
    mock.collection_exists = AsyncMock(return_value=True)
    mock.create_collection = AsyncMock(return_value=True)
    mock.upsert = AsyncMock(return_value=None)
    mock.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    return mock


def _stub_bm25(store: QdrantVectorStore, sparse: models.SparseVector | None = None) -> None:
    vector = sparse or models.SparseVector(indices=[1, 2], values=[0.5, 0.5])
    store._compute_bm25_sparse = lambda _text: vector  # type: ignore[method-assign]


def test_store_implements_protocol(store: QdrantVectorStore) -> None:
    assert isinstance(store, VectorStoreProtocol)


async def test_ensure_collection_creates_dense_and_sparse_vectors(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    await store.ensure_collection(vector_size=VECTOR_SIZE)

    info = await client.get_collection(COLLECTION)
    vectors = info.config.params.vectors
    sparse = info.config.params.sparse_vectors

    assert isinstance(vectors, dict)
    assert vectors[DENSE_VECTOR_NAME].size == VECTOR_SIZE
    assert vectors[DENSE_VECTOR_NAME].distance == models.Distance.COSINE
    assert sparse is not None
    assert sparse[SPARSE_VECTOR_NAME].modifier == models.Modifier.IDF


async def test_ensure_collection_is_idempotent(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    await store.ensure_collection(vector_size=VECTOR_SIZE)
    await store.ensure_collection(vector_size=VECTOR_SIZE)

    assert await client.collection_exists(COLLECTION)


async def test_ensure_collection_wraps_qdrant_error_in_value_error() -> None:
    mock_client = _mock_qdrant_client()
    mock_client.collection_exists.side_effect = RuntimeError("connection lost")
    store = QdrantVectorStore(mock_client, embedder=_mock_embedder(), collection_name=COLLECTION)

    with pytest.raises(ValueError, match="Failed to ensure Qdrant collection"):
        await store.ensure_collection(vector_size=VECTOR_SIZE)


async def test_upsert_writes_points_with_dense_and_bm25_vectors_and_payload(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    chunk = make_chunk("a" * 64)

    await store.upsert([chunk])

    points, _ = await client.scroll(COLLECTION, with_vectors=True, limit=10)
    assert len(points) == 1
    point = points[0]
    assert point.id == str(uuid.uuid5(POINT_ID_NAMESPACE, chunk.id))
    assert point.payload == {
        "chunk_id": chunk.id,
        "document_id": "doc_1",
        "text": chunk.text,
        "metadata": {"chunk_index": 0, "page": 3},
    }
    assert isinstance(point.vector, dict)
    assert point.vector[DENSE_VECTOR_NAME] == pytest.approx([0.6, 0.0, 0.8])
    sparse_vec = point.vector[SPARSE_VECTOR_NAME]
    assert isinstance(sparse_vec, models.SparseVector)
    assert len(sparse_vec.indices) > 0


async def test_upsert_overwrites_existing_chunk(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    await store.upsert([make_chunk("b" * 64)])
    updated = Chunk(
        id="b" * 64,
        document_id="doc_1",
        text="updated text",
        embedding=[0.3, 0.2, 0.1],
    )

    await store.upsert([updated])

    assert (await client.count(COLLECTION, exact=True)).count == 1
    points, _ = await client.scroll(COLLECTION, with_payload=True, limit=10)
    assert points[0].payload is not None
    assert points[0].payload["text"] == "updated text"


async def test_upsert_batch_of_chunks(store: QdrantVectorStore, client: AsyncQdrantClient) -> None:
    chunks = [make_chunk(f"{index:064d}") for index in range(5)]

    await store.upsert(chunks)

    assert (await client.count(COLLECTION, exact=True)).count == 5


async def test_upsert_empty_list_does_not_create_collection(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    await store.upsert([])

    assert not await client.collection_exists(COLLECTION)


async def test_upsert_computes_missing_embeddings_and_attaches_bm25_document() -> None:
    mock_client = _mock_qdrant_client()
    mock_emb = _mock_embedder(embed_batch_return=[[0.5, 0.5, 0.0]])
    store = QdrantVectorStore(mock_client, embedder=mock_emb, collection_name=COLLECTION)

    pre_embedded = Chunk(
        id="c1", document_id="doc_1", text="pre embedded", embedding=[0.1, 0.2, 0.3]
    )
    needs_embedding = Chunk(id="c2", document_id="doc_1", text="needs embedding", embedding=None)

    await store.upsert([pre_embedded, needs_embedding])

    mock_emb.embed_batch.assert_awaited_once_with(["needs embedding"])
    mock_client.upsert.assert_awaited_once()
    points = mock_client.upsert.call_args.kwargs["points"]
    assert len(points) == 2
    assert points[0].vector[DENSE_VECTOR_NAME] == [0.1, 0.2, 0.3]
    assert points[1].vector[DENSE_VECTOR_NAME] == [0.5, 0.5, 0.0]
    assert points[0].vector[SPARSE_VECTOR_NAME] == models.Document(
        text="pre embedded", model=BM25_MODEL
    )
    assert points[1].vector[SPARSE_VECTOR_NAME] == models.Document(
        text="needs embedding", model=BM25_MODEL
    )


async def test_upsert_raises_value_error_on_mismatched_embedder_count() -> None:
    mock_client = _mock_qdrant_client()
    mock_emb = _mock_embedder(embed_batch_return=[])
    store = QdrantVectorStore(mock_client, embedder=mock_emb, collection_name=COLLECTION)

    with pytest.raises(ValueError, match="Embedder returned 0 vectors for 1 chunks"):
        await store.upsert([Chunk(id="c1", document_id="doc_1", text="missing")])


async def test_upsert_wraps_embedder_and_qdrant_errors_in_value_error() -> None:
    mock_client = _mock_qdrant_client()
    failing_emb = _mock_embedder(embed_batch_side_effect=RuntimeError("ollama down"))
    store = QdrantVectorStore(mock_client, embedder=failing_emb, collection_name=COLLECTION)

    with pytest.raises(ValueError, match="Failed to embed chunks"):
        await store.upsert([Chunk(id="c1", document_id="doc_1", text="hello")])

    mock_client.upsert.side_effect = RuntimeError("qdrant down")
    ok_store = QdrantVectorStore(mock_client, embedder=_mock_embedder(), collection_name=COLLECTION)

    with pytest.raises(ValueError, match="Failed to upsert chunks into Qdrant"):
        await ok_store.upsert([make_chunk("d" * 64)])


async def test_hybrid_search_reuses_query_embedding_and_builds_rrf_prefetch() -> None:
    mock_client = _mock_qdrant_client()
    mock_emb = _mock_embedder()
    store = QdrantVectorStore(mock_client, embedder=mock_emb, collection_name=COLLECTION)
    bm25_vec = models.SparseVector(indices=[10, 20], values=[1.0, 0.5])
    _stub_bm25(store, bm25_vec)

    query = Query(text="hybrid retrieval", embedding=[0.9, 0.1, 0.0])
    await store.hybrid_search(query, top_k=7)

    mock_emb.embed_text.assert_not_awaited()
    mock_client.query_points.assert_awaited_once()
    kwargs = mock_client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == COLLECTION
    assert kwargs["limit"] == 7
    assert kwargs["with_payload"] is True
    assert kwargs["query"] == models.FusionQuery(fusion=models.Fusion.RRF)

    prefetch = kwargs["prefetch"]
    assert len(prefetch) == 2
    assert prefetch[0] == models.Prefetch(
        query=[0.9, 0.1, 0.0],
        using=DENSE_VECTOR_NAME,
        filter=None,
        limit=7,
    )
    assert prefetch[1] == models.Prefetch(
        query=bm25_vec,
        using=SPARSE_VECTOR_NAME,
        filter=None,
        limit=7,
    )


async def test_hybrid_search_generates_embedding_when_missing() -> None:
    mock_client = _mock_qdrant_client()
    mock_emb = _mock_embedder(embed_text_return=[0.4, 0.5, 0.6])
    store = QdrantVectorStore(mock_client, embedder=mock_emb, collection_name=COLLECTION)
    _stub_bm25(store)

    await store.hybrid_search(Query(text="semantic query"), top_k=5)

    mock_emb.embed_text.assert_awaited_once_with("semantic query")
    prefetch = mock_client.query_points.call_args.kwargs["prefetch"]
    assert prefetch[0].query == [0.4, 0.5, 0.6]


async def test_hybrid_search_forwards_filters_and_maps_scored_points() -> None:
    mock_client = _mock_qdrant_client()
    mock_client.query_points.return_value = SimpleNamespace(
        points=[
            models.ScoredPoint(
                id=str(uuid.uuid5(POINT_ID_NAMESPACE, "chunk-1")),
                version=1,
                score=0.875,
                payload={
                    "chunk_id": "chunk-1",
                    "document_id": "doc-42",
                    "text": "relevant text",
                    "metadata": {"page": 2, "author": "Ada"},
                },
            )
        ]
    )
    store = QdrantVectorStore(mock_client, embedder=_mock_embedder(), collection_name=COLLECTION)
    _stub_bm25(store)

    query = Query(
        text="filtered search",
        embedding=[0.1, 0.2, 0.3],
        filters={"document_id": "doc-42", "page": [1, 2], "metadata.author": "Ada"},
    )
    results = await store.hybrid_search(query, top_k=3)

    expected_filter = models.Filter(
        must=[
            models.FieldCondition(key="document_id", match=models.MatchValue(value="doc-42")),
            models.FieldCondition(key="metadata.page", match=models.MatchAny(any=[1, 2])),
            models.FieldCondition(key="metadata.author", match=models.MatchValue(value="Ada")),
        ]
    )
    prefetch = mock_client.query_points.call_args.kwargs["prefetch"]
    assert prefetch[0].filter == expected_filter
    assert prefetch[1].filter == expected_filter

    assert len(results) == 1
    assert results[0] == Chunk(
        id="chunk-1",
        document_id="doc-42",
        text="relevant text",
        metadata={"page": 2, "author": "Ada"},
        score=0.875,
    )


async def test_hybrid_search_empty_results_and_non_positive_top_k() -> None:
    mock_client = _mock_qdrant_client()
    mock_emb = _mock_embedder()
    store = QdrantVectorStore(mock_client, embedder=mock_emb, collection_name=COLLECTION)
    _stub_bm25(store)

    assert await store.hybrid_search(Query(text="nothing"), top_k=0) == []
    mock_emb.embed_text.assert_not_awaited()
    mock_client.query_points.assert_not_awaited()

    assert await store.hybrid_search(Query(text="empty"), top_k=5) == []


async def test_hybrid_search_wraps_errors_and_invalid_payloads() -> None:
    mock_client = _mock_qdrant_client()
    failing_emb = _mock_embedder(embed_text_side_effect=RuntimeError("embed failed"))
    store = QdrantVectorStore(mock_client, embedder=failing_emb, collection_name=COLLECTION)
    _stub_bm25(store)

    with pytest.raises(ValueError, match="Failed to embed query"):
        await store.hybrid_search(Query(text="boom"))

    mock_client.query_points.side_effect = RuntimeError("qdrant failed")
    ok_store = QdrantVectorStore(mock_client, embedder=_mock_embedder(), collection_name=COLLECTION)
    _stub_bm25(ok_store)
    with pytest.raises(ValueError, match="Qdrant hybrid search failed"):
        await ok_store.hybrid_search(Query(text="boom", embedding=[0.1, 0.2, 0.3]))

    mock_client.query_points.side_effect = None
    mock_client.query_points.return_value = SimpleNamespace(
        points=[models.ScoredPoint(id="1", version=1, score=0.5, payload={"chunk_id": "c1"})]
    )
    with pytest.raises(ValueError, match="missing payload field"):
        await ok_store.hybrid_search(Query(text="bad payload", embedding=[0.1, 0.2, 0.3]))

    def fail_bm25(_text: str) -> models.SparseVector:
        raise RuntimeError("bm25 failed")

    ok_store._compute_bm25_sparse = fail_bm25  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="Failed to compute BM25 query vector"):
        await ok_store.hybrid_search(Query(text="bm25 error", embedding=[0.1, 0.2, 0.3]))
