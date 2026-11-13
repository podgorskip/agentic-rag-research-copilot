"""Deterministic retrieval-stage fakes, for use in the test suite only.

Each fake records how it was called and can be told to fail, so a test can exercise both the
happy path and error handling without Qdrant or a cross-encoder model.
"""

from agentic_rag_copilot.domain.models import Chunk, Query
from agentic_rag_copilot.domain.protocols import RerankerProtocol, VectorStoreProtocol


class RecordingVectorStore(VectorStoreProtocol):
    """Returns configured candidates from hybrid_search and records each call."""

    def __init__(
        self,
        candidates: list[Chunk] | None = None,
        *,
        fail_with: Exception | None = None,
    ) -> None:
        self.candidates = list(candidates or [])
        self.fail_with = fail_with
        self.calls: list[tuple[Query, int]] = []

    async def upsert(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError("RecordingVectorStore is retrieval-only")

    async def hybrid_search(self, query: Query, top_k: int = 10) -> list[Chunk]:
        self.calls.append((query, top_k))
        if self.fail_with is not None:
            raise self.fail_with
        return self.candidates[:top_k]


class RecordingReranker(RerankerProtocol):
    """Returns configured results from rerank and records each call."""

    def __init__(
        self,
        results: list[Chunk] | None = None,
        *,
        fail_with: Exception | None = None,
    ) -> None:
        self.results = list(results or [])
        self.fail_with = fail_with
        self.calls: list[tuple[str, list[Chunk], int]] = []

    async def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        self.calls.append((query, list(chunks), top_k))
        if self.fail_with is not None:
            raise self.fail_with
        return self.results[:top_k]
