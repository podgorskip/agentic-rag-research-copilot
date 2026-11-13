import hashlib
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
from qdrant_client import AsyncQdrantClient, models
from testcontainers.community.qdrant import QdrantContainer

from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.domain.models import Chunk
from tests.mocks.ingestion import MockEmbedder

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
POINT_ID_NAMESPACE = uuid.UUID("6f1c2a8e-3b4d-4e5f-9a7b-1c2d3e4f5a6b")

QDRANT_IMAGE = "qdrant/qdrant:v1.13.6"
UNIT_VECTOR = [0.6, 0.0, 0.8]


@pytest.fixture(scope="session")
def qdrant_container() -> Iterator[QdrantContainer]:
    """Starts a throwaway Qdrant instance for the whole test session."""
    with QdrantContainer(QDRANT_IMAGE) as container:
        yield container


@pytest.fixture
async def client(qdrant_container: QdrantContainer) -> AsyncIterator[AsyncQdrantClient]:
    qdrant = qdrant_container.get_async_client()
    yield qdrant
    await qdrant.close()


@pytest.fixture
async def store(client: AsyncQdrantClient) -> AsyncIterator[QdrantVectorStore]:
    """Gives each test its own collection and drops it afterwards."""
    collection_name = f"test_{uuid.uuid4().hex}"
    yield QdrantVectorStore(
        client, embedder=MockEmbedder(dimension=len(UNIT_VECTOR)), collection_name=collection_name
    )
    if await client.collection_exists(collection_name):
        await client.delete_collection(collection_name)


def make_chunk(
    index: int,
    document_id: str = "doc_1",
    text: str | None = None,
    metadata: dict[str, object] | None = None,
    embedding: list[float] | None = None,
) -> Chunk:
    return Chunk(
        id=hashlib.sha256(f"{document_id}_{index}".encode()).hexdigest(),
        document_id=document_id,
        text=text or f"dummy chunk {index}",
        metadata=metadata if metadata is not None else {"chunk_index": index, "page": index + 1},
        embedding=UNIT_VECTOR if embedding is None else embedding,
    )


def point_id_for(chunk: Chunk) -> str:
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk.id))


async def test_upsert_creates_collection_with_dense_and_sparse_vectors(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    await store.upsert([make_chunk(0)])

    info = await client.get_collection(store.collection_name)
    vectors = info.config.params.vectors
    sparse = info.config.params.sparse_vectors

    assert isinstance(vectors, dict)
    assert vectors[DENSE_VECTOR_NAME].size == len(UNIT_VECTOR)
    assert vectors[DENSE_VECTOR_NAME].distance == models.Distance.COSINE
    assert sparse is not None
    assert sparse[SPARSE_VECTOR_NAME].modifier == models.Modifier.IDF


async def test_upsert_persists_every_chunk_as_a_point(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    chunks = [make_chunk(index) for index in range(5)]

    await store.upsert(chunks)

    assert (await client.count(store.collection_name, exact=True)).count == len(chunks)
    points, _ = await client.scroll(store.collection_name, with_payload=True, limit=10)
    assert {point.id for point in points} == {point_id_for(chunk) for chunk in chunks}


async def test_upsert_stores_text_and_metadata_in_payload(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    chunk = make_chunk(
        0,
        text="Hybrid search combines BM25 and dense vectors. Zażółć gęślą jaźń.",
        metadata={"chunk_index": 0, "page": 3, "author": "Ada", "reviewed": True, "tags": ["ir"]},
    )

    await store.upsert([chunk])

    point = (
        await client.retrieve(store.collection_name, ids=[point_id_for(chunk)], with_payload=True)
    )[0]
    assert point.payload == {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "metadata": {
            "chunk_index": 0,
            "page": 3,
            "author": "Ada",
            "reviewed": True,
            "tags": ["ir"],
        },
    }


async def test_upsert_stores_dense_embedding_for_each_point(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    chunk = make_chunk(0, embedding=UNIT_VECTOR)

    await store.upsert([chunk])

    point = (
        await client.retrieve(store.collection_name, ids=[point_id_for(chunk)], with_vectors=True)
    )[0]
    assert isinstance(point.vector, dict)
    assert point.vector[DENSE_VECTOR_NAME] == pytest.approx(UNIT_VECTOR)


async def test_metadata_is_queryable_through_payload_filter(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    chunks = [make_chunk(index, metadata={"page": index % 2}) for index in range(4)]
    await store.upsert(chunks)

    page_zero, _ = await client.scroll(
        store.collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="metadata.page", match=models.MatchValue(value=0))]
        ),
        with_payload=True,
        limit=10,
    )

    expected = {point_id_for(chunk) for chunk in chunks if chunk.metadata["page"] == 0}
    assert {point.id for point in page_zero} == expected


async def test_upsert_preserves_document_ids_across_documents(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    chunks = [make_chunk(0, document_id="doc_a"), make_chunk(0, document_id="doc_b")]

    await store.upsert(chunks)

    by_document, _ = await client.scroll(
        store.collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value="doc_b"))]
        ),
        with_payload=True,
        limit=10,
    )
    assert len(by_document) == 1
    assert by_document[0].payload is not None
    assert by_document[0].payload["chunk_id"] == chunks[1].id


async def test_reupserting_a_chunk_updates_it_in_place(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    original = make_chunk(0, text="original text")
    await store.upsert([original])

    revised = make_chunk(0, text="revised text", metadata={"chunk_index": 0, "page": 9})
    await store.upsert([revised])

    assert (await client.count(store.collection_name, exact=True)).count == 1
    point = (
        await client.retrieve(store.collection_name, ids=[point_id_for(revised)], with_payload=True)
    )[0]
    assert point.payload is not None
    assert point.payload["text"] == "revised text"
    assert point.payload["metadata"] == {"chunk_index": 0, "page": 9}


async def test_upsert_empty_list_leaves_database_untouched(
    store: QdrantVectorStore, client: AsyncQdrantClient
) -> None:
    await store.upsert([])

    assert not await client.collection_exists(store.collection_name)
