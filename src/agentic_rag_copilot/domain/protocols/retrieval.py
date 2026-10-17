from typing import Protocol, runtime_checkable

from ..models.entities import Chunk, Query


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Contract for embedding models."""

    async def embed_text(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Contract for Qdrant hybrid search."""

    async def upsert(self, chunks: list[Chunk]) -> None: ...

    async def hybrid_search(self, query: Query, top_k: int = 10) -> list[Chunk]: ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Contract for bge-reranker-v2 cross-encoder."""

    async def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]: ...
