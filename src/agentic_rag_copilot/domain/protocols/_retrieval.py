from typing import Protocol, runtime_checkable

from ..models import Chunk, Query


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Contract for embedding models."""

    async def embed_text(self, text: str) -> list[float]:
        """Generate vector embedding for a single text string.

        Args:
            text: Text content to embed.

        Returns:
            Dense vector embedding representing the text.
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of dense vector embeddings corresponding to input texts.
        """
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Contract for Qdrant hybrid search."""

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks in the vector store.

        Args:
            chunks: List of Chunk entities to upsert.

        Returns:
            None.
        """
        ...

    async def hybrid_search(self, query: Query, top_k: int = 10) -> list[Chunk]:
        """Perform hybrid retrieval for a query.

        Args:
            query: Query model containing text, filters, and optional embedding.
            top_k: Number of most relevant chunks to retrieve.

        Returns:
            Top-k retrieved chunks matching the query.
        """
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Contract for bge-reranker-v2 cross-encoder."""

    async def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """Rerank retrieved chunks using a cross-encoder model.

        Args:
            query: Search query string.
            chunks: Candidate chunks to be scored and reordered.
            top_k: Number of highest-scoring chunks to return.

        Returns:
            Top-k reranked chunks with updated relevance scores.
        """
        ...
