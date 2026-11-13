import asyncio
from typing import Final

from fastembed.rerank.cross_encoder import TextCrossEncoder

from agentic_rag_copilot.domain.models import Chunk
from agentic_rag_copilot.domain.protocols import RerankerProtocol
from agentic_rag_copilot.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL: Final[str] = "jinaai/jina-reranker-v2-base-multilingual"
_RERANK_SCORE_KEY: Final[str] = "rerank_score"


class FastEmbedReranker(RerankerProtocol):
    """Cross-encoder reranker backed by FastEmbed's ONNX ``TextCrossEncoder``."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            model_name: FastEmbed-supported cross-encoder model identifier.
        """
        self._model_name = model_name
        self._encoder: TextCrossEncoder | None = None

    async def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """Rerank retrieved chunks using the configured cross-encoder model.

        Args:
            query: Search query string.
            chunks: Candidate chunks to be scored and reordered.
            top_k: Number of highest-scoring chunks to return.

        Returns:
            Top-k reranked chunks ordered by descending relevance, with ``score`` and
            ``metadata["rerank_score"]`` populated.

        Raises:
            ValueError: If model loading or cross-encoder inference fails.
        """
        if not chunks or top_k <= 0:
            return []

        texts = [chunk.text for chunk in chunks]
        try:
            scores = await asyncio.to_thread(self._score_pairs, query, texts)
        except Exception as exc:
            logger.error(f"Cross-encoder reranking failed with model '{self._model_name}': {exc}")
            raise ValueError(f"Cross-encoder reranking failed: {exc}") from exc

        if len(scores) != len(chunks):
            raise ValueError(
                f"Cross-encoder returned {len(scores)} scores for {len(chunks)} chunks"
            )

        scored = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            chunk.model_copy(
                update={
                    "score": score,
                    "metadata": {**chunk.metadata, _RERANK_SCORE_KEY: score},
                }
            )
            for chunk, score in scored[:top_k]
        ]

    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Synchronously load the cross-encoder if needed and score query-text pairs.

        Args:
            query: Search query string.
            texts: Candidate chunk texts aligned with the input chunks.

        Returns:
            Relevance scores as floats aligned with ``texts``.
        """
        if self._encoder is None:
            self._encoder = TextCrossEncoder(model_name=self._model_name)
        return [float(score) for score in self._encoder.rerank(query=query, documents=texts)]
