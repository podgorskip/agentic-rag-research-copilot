from typing import cast
from unittest.mock import MagicMock, create_autospec

from langfuse import Langfuse

from agentic_rag_copilot.domain.models import Chunk, Query
from agentic_rag_copilot.services import RetrievalService
from tests.mocks.ingestion import MockEmbedder
from tests.mocks.retrieval import RecordingReranker, RecordingVectorStore


def _make_chunk(chunk_id: str, score: float | None = None) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc-1", text=f"text for {chunk_id}", score=score)


def _fake_langfuse() -> MagicMock:
    return create_autospec(Langfuse, instance=True)


def build_service(
    embedder: MockEmbedder | None = None,
    vector_store: RecordingVectorStore | None = None,
    reranker: RecordingReranker | None = None,
    langfuse_client: MagicMock | None = None,
) -> RetrievalService:
    return RetrievalService(
        embedder=embedder or MockEmbedder(),
        vector_store=vector_store or RecordingVectorStore(),
        reranker=reranker or RecordingReranker(),
        langfuse_client=cast("Langfuse", langfuse_client or _fake_langfuse()),
    )


async def test_retrieve_embeds_query_when_embedding_missing_and_forwards_params() -> None:
    embedder = MockEmbedder(dimension=3)
    candidates = [_make_chunk("c1")]
    results = [_make_chunk("c1", score=0.9)]
    vector_store = RecordingVectorStore(candidates)
    reranker = RecordingReranker(results)
    service = build_service(embedder, vector_store, reranker)
    expected_embedding = await MockEmbedder(dimension=3).embed_text("what is RAG?")

    result = await service.retrieve(Query(text="what is RAG?"), initial_top_k=10, final_top_k=3)

    assert result == results
    assert len(embedder.batches) == 1
    searched_query, top_k = vector_store.calls[0]
    assert searched_query.text == "what is RAG?"
    assert searched_query.embedding == expected_embedding
    assert top_k == 10
    rerank_query, rerank_candidates, rerank_top_k = reranker.calls[0]
    assert rerank_query == "what is RAG?"
    assert rerank_candidates == candidates
    assert rerank_top_k == 3


async def test_retrieve_skips_embedding_when_already_set() -> None:
    embedder = MockEmbedder()
    vector_store = RecordingVectorStore([_make_chunk("c1")])
    reranker = RecordingReranker([_make_chunk("c1", score=0.5)])
    service = build_service(embedder, vector_store, reranker)
    query = Query(text="q", embedding=[1.0, 0.0, 0.0])

    await service.retrieve(query)

    assert embedder.batches == []
    searched_query, _ = vector_store.calls[0]
    assert searched_query.embedding == [1.0, 0.0, 0.0]


async def test_retrieve_short_circuits_on_empty_candidates_without_reranking() -> None:
    reranker = RecordingReranker([_make_chunk("c1")])
    service = build_service(vector_store=RecordingVectorStore([]), reranker=reranker)

    result = await service.retrieve(Query(text="q", embedding=[0.0]))

    assert result == []
    assert reranker.calls == []


async def test_retrieve_uses_default_top_k_values_when_not_specified() -> None:
    vector_store = RecordingVectorStore([_make_chunk(f"c{i}") for i in range(25)])
    reranker = RecordingReranker([_make_chunk("c0", score=1.0)])
    service = build_service(vector_store=vector_store, reranker=reranker)

    await service.retrieve(Query(text="q", embedding=[0.0]))

    _, initial_top_k = vector_store.calls[0]
    _, _, final_top_k = reranker.calls[0]
    assert initial_top_k == 20
    assert final_top_k == 5


async def test_retrieve_instruments_hybrid_search_and_rerank_spans() -> None:
    langfuse_client = _fake_langfuse()
    vector_store = RecordingVectorStore([_make_chunk("c1"), _make_chunk("c2")])
    reranker = RecordingReranker([_make_chunk("c1", score=0.77)])
    service = build_service(
        vector_store=vector_store, reranker=reranker, langfuse_client=langfuse_client
    )

    await service.retrieve(Query(text="q", embedding=[0.0]), initial_top_k=2, final_top_k=1)

    assert langfuse_client.start_as_current_span.call_count == 2
    hybrid_call, rerank_call = langfuse_client.start_as_current_span.call_args_list
    assert hybrid_call.kwargs["name"] == "hybrid_search"
    assert hybrid_call.kwargs["input"] == {"query": "q", "top_k": 2}
    assert rerank_call.kwargs["name"] == "rerank"
    assert rerank_call.kwargs["input"] == {"query": "q", "candidate_count": 2, "top_k": 1}

    span = langfuse_client.start_as_current_span.return_value.__enter__.return_value
    hybrid_update, rerank_update = span.update.call_args_list

    assert hybrid_update.kwargs["output"] == {"candidate_count": 2}
    assert hybrid_update.kwargs["metadata"]["candidate_count"] == 2
    assert hybrid_update.kwargs["metadata"]["latency_ms"] >= 0

    assert rerank_update.kwargs["output"] == {"result_count": 1, "top_score": 0.77}
    assert rerank_update.kwargs["metadata"]["result_count"] == 1
    assert rerank_update.kwargs["metadata"]["top_score"] == 0.77
    assert rerank_update.kwargs["metadata"]["latency_ms"] >= 0


async def test_retrieve_records_no_top_score_when_reranker_returns_nothing() -> None:
    langfuse_client = _fake_langfuse()
    service = build_service(
        vector_store=RecordingVectorStore([_make_chunk("c1")]),
        reranker=RecordingReranker([]),
        langfuse_client=langfuse_client,
    )

    result = await service.retrieve(Query(text="q", embedding=[0.0]))

    assert result == []
    span = langfuse_client.start_as_current_span.return_value.__enter__.return_value
    _, rerank_update = span.update.call_args_list
    assert rerank_update.kwargs["output"]["top_score"] is None
