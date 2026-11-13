from unittest.mock import MagicMock

import pytest

from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker
from agentic_rag_copilot.domain.models import Chunk
from agentic_rag_copilot.domain.protocols import RerankerProtocol


def _make_chunk(chunk_id: str, text: str, **metadata: object) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-1",
        text=text,
        metadata=dict(metadata),
        embedding=[0.1, 0.2],
        score=0.5,
    )


def test_reranker_implements_protocol() -> None:
    reranker = FastEmbedReranker()
    assert isinstance(reranker, RerankerProtocol)


async def test_rerank_orders_descending_slices_top_k_and_preserves_metadata() -> None:
    reranker = FastEmbedReranker()
    chunks = [
        _make_chunk("c1", "distractor one", page=1, source="a.pdf"),
        _make_chunk("c2", "exact answer", page=2, source="b.pdf"),
        _make_chunk("c3", "partial match", page=3, source="c.pdf"),
    ]
    mock_encoder = MagicMock()
    mock_encoder.rerank.return_value = [0.15, 0.92, 0.48]
    reranker._encoder = mock_encoder  # type: ignore[assignment]

    result = await reranker.rerank("query", chunks, top_k=2)

    mock_encoder.rerank.assert_called_once_with(
        query="query",
        documents=["distractor one", "exact answer", "partial match"],
    )
    assert [chunk.id for chunk in result] == ["c2", "c3"]
    assert result[0].document_id == "doc-1"
    assert result[0].text == "exact answer"
    assert result[0].embedding == [0.1, 0.2]
    assert result[0].score == pytest.approx(0.92)
    assert result[0].metadata == {"page": 2, "source": "b.pdf", "rerank_score": 0.92}
    assert result[1].metadata == {"page": 3, "source": "c.pdf", "rerank_score": 0.48}
    # Original frozen chunks must remain untouched.
    assert chunks[1].score == 0.5
    assert "rerank_score" not in chunks[1].metadata


async def test_rerank_when_fewer_chunks_than_top_k_returns_all_sorted() -> None:
    reranker = FastEmbedReranker()
    chunks = [_make_chunk("c1", "first"), _make_chunk("c2", "second")]
    mock_encoder = MagicMock()
    mock_encoder.rerank.return_value = [0.2, 0.8]
    reranker._encoder = mock_encoder  # type: ignore[assignment]

    result = await reranker.rerank("query", chunks, top_k=5)

    assert [chunk.id for chunk in result] == ["c2", "c1"]


async def test_rerank_short_circuits_on_empty_chunks_or_non_positive_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = MagicMock()
    monkeypatch.setattr(
        "agentic_rag_copilot.adapters.retrieval._fastembed_reranker.TextCrossEncoder",
        loader,
    )
    reranker = FastEmbedReranker()

    assert await reranker.rerank("query", [], top_k=5) == []
    assert await reranker.rerank("query", [_make_chunk("c1", "text")], top_k=0) == []
    assert await reranker.rerank("query", [_make_chunk("c1", "text")], top_k=-2) == []
    loader.assert_not_called()


async def test_rerank_wraps_model_loading_and_inference_errors_in_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agentic_rag_copilot.adapters.retrieval._fastembed_reranker.TextCrossEncoder",
        MagicMock(side_effect=RuntimeError("model download failed")),
    )
    reranker = FastEmbedReranker()

    with pytest.raises(ValueError, match="Cross-encoder reranking failed"):
        await reranker.rerank("query", [_make_chunk("c1", "text")], top_k=1)

    failing_encoder = MagicMock()
    failing_encoder.rerank.side_effect = RuntimeError("onnx inference failed")
    reranker._encoder = failing_encoder  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Cross-encoder reranking failed"):
        await reranker.rerank("query", [_make_chunk("c1", "text")], top_k=1)


async def test_rerank_raises_value_error_on_mismatched_score_count() -> None:
    reranker = FastEmbedReranker()
    mock_encoder = MagicMock()
    mock_encoder.rerank.return_value = [0.9]
    reranker._encoder = mock_encoder  # type: ignore[assignment]

    with pytest.raises(ValueError, match="returned 1 scores for 2 chunks"):
        await reranker.rerank("query", [_make_chunk("c1", "a"), _make_chunk("c2", "b")], top_k=2)
