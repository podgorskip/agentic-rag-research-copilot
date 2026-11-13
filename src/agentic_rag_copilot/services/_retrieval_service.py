import time
from typing import Final

from langfuse import Langfuse, get_client

from agentic_rag_copilot.domain.models import Chunk, Query
from agentic_rag_copilot.domain.protocols import (
    EmbedderProtocol,
    RerankerProtocol,
    VectorStoreProtocol,
)

_DEFAULT_INITIAL_TOP_K: Final[int] = 20
_DEFAULT_FINAL_TOP_K: Final[int] = 5


class RetrievalService:
    """Two-stage retrieval: broad hybrid candidates, refined via cross-encoder reranking."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        reranker: RerankerProtocol,
        langfuse_client: Langfuse | None = None,
    ) -> None:
        """Initialize the retrieval service.

        Args:
            embedder: Embedder used to populate a query's dense vector when missing.
            vector_store: Hybrid (dense + BM25) candidate retriever.
            reranker: Cross-encoder used to refine candidates down to the final top-k.
            langfuse_client: Langfuse client used to emit stage spans. Defaults to the
                process-wide client from ``langfuse.get_client()``.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._langfuse = langfuse_client if langfuse_client is not None else get_client()

    async def retrieve(
        self,
        query: Query,
        initial_top_k: int = _DEFAULT_INITIAL_TOP_K,
        final_top_k: int = _DEFAULT_FINAL_TOP_K,
    ) -> list[Chunk]:
        """Retrieve the most relevant chunks for a query via hybrid search then reranking.

        Args:
            query: Search query. Its ``embedding`` is computed via the embedder if absent.
            initial_top_k: Number of broad candidates to fetch from the vector store.
            final_top_k: Number of highest-relevance chunks to return after reranking.

        Returns:
            Up to ``final_top_k`` reranked chunks, or an empty list if hybrid search finds
            no candidates.
        """
        if query.embedding is None:
            embedding = await self._embedder.embed_text(query.text)
            query = query.model_copy(update={"embedding": embedding})

        candidates = await self._traced_hybrid_search(query, initial_top_k)
        if not candidates:
            return []

        return await self._traced_rerank(query.text, candidates, final_top_k)

    async def _traced_hybrid_search(self, query: Query, top_k: int) -> list[Chunk]:
        """Run hybrid search inside a Langfuse span recording latency and candidate count.

        Args:
            query: Search query with its embedding already resolved.
            top_k: Number of candidates to fetch.

        Returns:
            Candidates returned by the vector store.
        """
        with self._langfuse.start_as_current_span(
            name="hybrid_search",
            input={"query": query.text, "top_k": top_k},
        ) as span:
            start = time.perf_counter()
            candidates = await self._vector_store.hybrid_search(query, top_k=top_k)
            latency_ms = (time.perf_counter() - start) * 1000
            span.update(
                output={"candidate_count": len(candidates)},
                metadata={"latency_ms": latency_ms, "candidate_count": len(candidates)},
            )
        return candidates

    async def _traced_rerank(
        self, query_text: str, candidates: list[Chunk], top_k: int
    ) -> list[Chunk]:
        """Run reranking inside a Langfuse span recording latency, counts, and top score.

        Args:
            query_text: Original query text.
            candidates: Hybrid search candidates to rerank.
            top_k: Number of highest-relevance chunks to keep.

        Returns:
            Reranked chunks returned by the reranker.
        """
        with self._langfuse.start_as_current_span(
            name="rerank",
            input={"query": query_text, "candidate_count": len(candidates), "top_k": top_k},
        ) as span:
            start = time.perf_counter()
            reranked = await self._reranker.rerank(query_text, candidates, top_k=top_k)
            latency_ms = (time.perf_counter() - start) * 1000
            top_score = reranked[0].score if reranked else None
            span.update(
                output={"result_count": len(reranked), "top_score": top_score},
                metadata={
                    "latency_ms": latency_ms,
                    "result_count": len(reranked),
                    "top_score": top_score,
                },
            )
        return reranked
