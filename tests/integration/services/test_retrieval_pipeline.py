from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import MagicMock, create_autospec

import pytest
from langfuse import Langfuse
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker
from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.domain.models import Chunk, Query
from agentic_rag_copilot.domain.protocols import EmbedderProtocol
from agentic_rag_copilot.services import RetrievalService


class _SemanticTestEmbedder(EmbedderProtocol):
    """Deterministic embedder separating retrieval concepts onto distinct axes."""

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    @staticmethod
    def _encode(text: str) -> list[float]:
        lowered = text.lower()
        if any(term in lowered for term in ("rrf", "reciprocal", "fusion", "bm25", "hybrid")):
            return [1.0, 0.0, 0.0]
        if any(term in lowered for term in ("qdrant", "vector", "collection")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.fixture
async def qdrant_client() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


@pytest.fixture
def pipeline() -> tuple[RetrievalService, QdrantVectorStore]:
    raise NotImplementedError("Use retrieval_pipeline fixture with qdrant_client")


@pytest.fixture
def retrieval_pipeline(
    qdrant_client: AsyncQdrantClient,
) -> tuple[RetrievalService, QdrantVectorStore]:
    embedder = _SemanticTestEmbedder()
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        embedder=embedder,
        collection_name="pipeline_integration_test",
    )
    reranker = FastEmbedReranker()
    langfuse_stub = cast("Langfuse", cast("MagicMock", create_autospec(Langfuse, instance=True)))
    service = RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        langfuse_client=langfuse_stub,
    )
    return service, vector_store


async def _seed_corpus(vector_store: QdrantVectorStore) -> None:
    chunks = [
        Chunk(
            id="c-exact-rrf",
            document_id="doc-rag",
            text=(
                "Reciprocal Rank Fusion (RRF) merges dense cosine similarity rankings with "
                "sparse BM25 lexical rankings without requiring score normalization."
            ),
            metadata={"source": "rag_paper.pdf", "page": 4},
        ),
        Chunk(
            id="c-partial-hybrid",
            document_id="doc-rag",
            text="Hybrid search queries both dense and sparse indexes in a single request.",
            metadata={"source": "rag_paper.pdf", "page": 5},
        ),
        Chunk(
            id="c-qdrant-ops",
            document_id="doc-qdrant",
            text="Qdrant collections store named dense vectors and IDF-weighted sparse vectors.",
            metadata={"source": "qdrant_docs.html", "page": 1},
        ),
        Chunk(
            id="c-distractor",
            document_id="doc-baking",
            text="Sourdough fermentation relies on wild yeast and lactic acid bacteria.",
            metadata={"source": "baking.html", "page": 2},
        ),
    ]
    await vector_store.upsert(chunks)


async def test_end_to_end_retrieval_pipeline_ranks_relevant_chunk_first(
    retrieval_pipeline: tuple[RetrievalService, QdrantVectorStore],
) -> None:
    service, vector_store = retrieval_pipeline
    await _seed_corpus(vector_store)

    results = await service.retrieve(
        Query(text="How does Reciprocal Rank Fusion combine dense and BM25 rankings?"),
        initial_top_k=4,
        final_top_k=2,
    )

    assert len(results) == 2
    assert results[0].id == "c-exact-rrf"
    assert results[0].score is not None
    assert "rerank_score" in results[0].metadata
    assert results[0].metadata["source"] == "rag_paper.pdf"
    assert results[0].metadata["page"] == 4
    assert results[0].score >= (results[1].score or float("-inf"))


async def test_end_to_end_retrieval_pipeline_respects_query_filters(
    retrieval_pipeline: tuple[RetrievalService, QdrantVectorStore],
) -> None:
    service, vector_store = retrieval_pipeline
    await _seed_corpus(vector_store)

    results = await service.retrieve(
        Query(
            text="How does Reciprocal Rank Fusion work?",
            filters={"document_id": "doc-qdrant"},
        ),
        initial_top_k=4,
        final_top_k=2,
    )

    assert len(results) == 1
    assert results[0].id == "c-qdrant-ops"
    assert results[0].document_id == "doc-qdrant"
    assert "rerank_score" in results[0].metadata


async def test_end_to_end_retrieval_pipeline_returns_empty_when_no_candidates(
    retrieval_pipeline: tuple[RetrievalService, QdrantVectorStore],
) -> None:
    service, vector_store = retrieval_pipeline
    await vector_store.ensure_collection(vector_size=3)

    assert await service.retrieve(Query(text="nonexistent topic")) == []

    await _seed_corpus(vector_store)
    filtered = await service.retrieve(
        Query(text="Reciprocal Rank Fusion", filters={"document_id": "nonexistent-doc"}),
        initial_top_k=5,
        final_top_k=3,
    )
    assert filtered == []
