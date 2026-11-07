import uuid
from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient, models

from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.adapters.vector_store._qdrant_store import (
    DENSE_VECTOR_NAME,
    POINT_ID_NAMESPACE,
    SPARSE_VECTOR_NAME,
)
from agentic_rag_copilot.domain.models import Chunk, Query
from agentic_rag_copilot.domain.protocols import VectorStoreProtocol

COLLECTION = "test_chunks"
VECTOR_SIZE = 3


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    qdrant = AsyncQdrantClient(":memory:")
    yield qdrant
    await qdrant.close()


@pytest.fixture
def store(client: AsyncQdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(client, collection_name=COLLECTION)


def make_chunk(chunk_id: str, embedding: list[float] | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc_1",
        text=f"text for {chunk_id}",
        metadata={"chunk_index": 0, "page": 3},
        embedding=[0.6, 0.0, 0.8] if embedding is None else embedding,
    )


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


async def test_upsert_writes_points_with_dense_vector_and_payload(
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


async def test_upsert_rejects_chunk_without_embedding(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    chunk = Chunk(id="c" * 64, document_id="doc_1", text="no vector")

    with pytest.raises(ValueError, match="has no embedding"):
        await store.upsert([chunk])

    assert not await client.collection_exists(COLLECTION)


async def test_hybrid_search_is_not_implemented_yet(store: QdrantVectorStore) -> None:
    query = Query(text="what is bm25")

    with pytest.raises(NotImplementedError):
        await store.hybrid_search(query)
