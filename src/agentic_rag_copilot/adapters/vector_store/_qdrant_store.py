import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from agentic_rag_copilot.domain.models import Chunk, Query
from agentic_rag_copilot.domain.protocols import VectorStoreProtocol

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
POINT_ID_NAMESPACE = uuid.UUID("6f1c2a8e-3b4d-4e5f-9a7b-1c2d3e4f5a6b")


class QdrantVectorStore(VectorStoreProtocol):
    """Qdrant-backed store holding dense and sparse (BM25) vectors per chunk."""

    def __init__(self, client: AsyncQdrantClient, collection_name: str = "chunks") -> None:
        self.client = client
        self.collection_name = collection_name

    async def ensure_collection(self, vector_size: int) -> None:
        """Create the collection with named dense and sparse vectors if it is missing.

        Args:
            vector_size: Dimensionality of the dense embeddings.

        Returns:
            None.
        """
        if await self.client.collection_exists(self.collection_name):
            return

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks in the Qdrant collection.

        Args:
            chunks: List of Chunk entities to upsert. Every chunk must carry an embedding.

        Returns:
            None.

        Raises:
            ValueError: If a chunk has no dense embedding.
        """
        if not chunks:
            return

        embeddings: list[list[float]] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk '{chunk.id}' has no embedding and cannot be upserted")
            embeddings.append(chunk.embedding)

        await self.ensure_collection(vector_size=len(embeddings[0]))

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(POINT_ID_NAMESPACE, chunk.id)),
                vector={DENSE_VECTOR_NAME: embedding},
                payload=self._to_payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def hybrid_search(self, query: Query, top_k: int = 10) -> list[Chunk]:
        """Perform hybrid retrieval for a query.

        Args:
            query: Query model containing text, filters, and optional embedding.
            top_k: Number of most relevant chunks to retrieve.

        Returns:
            Top-k retrieved chunks matching the query.

        Raises:
            NotImplementedError: Hybrid search is not implemented yet.
        """
        raise NotImplementedError("Hybrid search is not implemented yet")

    @staticmethod
    def _to_payload(chunk: Chunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
