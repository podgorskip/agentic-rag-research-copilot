from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.domain.models import Chunk, Query
from agentic_rag_copilot.domain.protocols import EmbedderProtocol


class _KeywordAwareEmbedder(EmbedderProtocol):
    """Deterministic embedder mapping semantic concepts to distinct orthogonal axes."""

    def __init__(self) -> None:
        self.embed_text_calls: list[str] = []

    async def embed_text(self, text: str) -> list[float]:
        self.embed_text_calls.append(text)
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    @staticmethod
    def _encode(text: str) -> list[float]:
        lowered = text.lower()
        if any(term in lowered for term in ("attention", "transformer", "self-attention")):
            return [1.0, 0.0, 0.0]
        if any(term in lowered for term in ("neural", "weight", "gradient")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    qdrant = AsyncQdrantClient(":memory:")
    yield qdrant
    await qdrant.close()


@pytest.fixture
def embedder() -> _KeywordAwareEmbedder:
    return _KeywordAwareEmbedder()


@pytest.fixture
def store(client: AsyncQdrantClient, embedder: _KeywordAwareEmbedder) -> QdrantVectorStore:
    return QdrantVectorStore(client, embedder=embedder, collection_name="hybrid_test")


async def _seed_chunks(store: QdrantVectorStore) -> None:
    chunks = [
        Chunk(
            id="chunk-bm25",
            document_id="doc-net",
            text="RFC 9110 defines HTTP semantics and status codes for QUIC transport.",
            metadata={"topic": "networking", "page": 1},
        ),
        Chunk(
            id="chunk-dense",
            document_id="doc-ml",
            text="Multi-head self-attention allows transformers to attend across positions.",
            metadata={"topic": "ml", "page": 2},
        ),
        Chunk(
            id="chunk-other",
            document_id="doc-ml",
            text="Stochastic gradient descent updates neural network weights iteratively.",
            metadata={"topic": "ml", "page": 3},
        ),
    ]
    await store.upsert(chunks)


async def test_hybrid_search_ranks_exact_keyword_match_first_via_bm25(
    store: QdrantVectorStore,
) -> None:
    await _seed_chunks(store)

    results = await store.hybrid_search(Query(text="RFC 9110 QUIC"), top_k=3)

    assert results
    assert results[0].id == "chunk-bm25"
    assert results[0].score is not None
    assert results[0].metadata == {"topic": "networking", "page": 1}


async def test_hybrid_search_ranks_conceptual_paraphrase_first_via_dense(
    store: QdrantVectorStore,
) -> None:
    await _seed_chunks(store)

    results = await store.hybrid_search(
        Query(text="how does the transformer attention mechanism work"),
        top_k=3,
    )

    assert results
    assert results[0].id == "chunk-dense"
    assert results[0].score is not None


async def test_hybrid_search_applies_metadata_and_document_filters(
    store: QdrantVectorStore,
) -> None:
    await _seed_chunks(store)

    results = await store.hybrid_search(
        Query(text="RFC 9110", filters={"document_id": "doc-ml", "page": [2]}),
        top_k=3,
    )

    assert len(results) == 1
    assert results[0].id == "chunk-dense"
    assert results[0].document_id == "doc-ml"
    assert results[0].metadata == {"topic": "ml", "page": 2}


async def test_hybrid_search_reuses_precomputed_query_embedding(
    store: QdrantVectorStore, embedder: _KeywordAwareEmbedder
) -> None:
    await _seed_chunks(store)

    results = await store.hybrid_search(
        Query(text="unrelated query text", embedding=[0.0, 1.0, 0.0]),
        top_k=2,
    )

    assert embedder.embed_text_calls == []
    assert len(results) == 2
    assert results[0].id == "chunk-other"


async def test_hybrid_search_respects_top_k_limit(store: QdrantVectorStore) -> None:
    await _seed_chunks(store)

    results = await store.hybrid_search(Query(text="network weights"), top_k=1)

    assert len(results) == 1
    assert results[0].score is not None
