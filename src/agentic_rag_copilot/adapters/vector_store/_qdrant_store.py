import asyncio
import uuid
from typing import Any, Final, cast

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models

from agentic_rag_copilot.domain.models import Chunk, Query
from agentic_rag_copilot.domain.protocols import EmbedderProtocol, VectorStoreProtocol
from agentic_rag_copilot.logging import get_logger

logger = get_logger(__name__)

_DENSE_VECTOR_NAME: Final[str] = "dense"
_SPARSE_VECTOR_NAME: Final[str] = "bm25"
_BM25_MODEL: Final[str] = "Qdrant/bm25"
_POINT_ID_NAMESPACE: Final[uuid.UUID] = uuid.UUID("6f1c2a8e-3b4d-4e5f-9a7b-1c2d3e4f5a6b")
_TOP_LEVEL_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset({"chunk_id", "document_id", "text"})


class QdrantVectorStore(VectorStoreProtocol):
    """Qdrant-backed store holding dense and sparse (BM25) vectors per chunk."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: EmbedderProtocol,
        collection_name: str = "chunks",
    ) -> None:
        """Initialize the Qdrant hybrid vector store.

        Args:
            client: Async Qdrant client instance.
            embedder: Embedder used to compute dense vectors when not pre-populated.
            collection_name: Name of the target Qdrant collection.
        """
        self.client = client
        self._embedder = embedder
        self.collection_name = collection_name
        self._bm25_embedder: SparseTextEmbedding | None = None

    async def ensure_collection(self, vector_size: int) -> None:
        """Create the collection with named dense and sparse vectors if it is missing.

        Args:
            vector_size: Dimensionality of the dense embeddings.

        Returns:
            None.

        Raises:
            ValueError: If Qdrant fails while checking or creating the collection.
        """
        try:
            if await self.client.collection_exists(self.collection_name):
                return

            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    _DENSE_VECTOR_NAME: models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    _SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
            )
        except Exception as exc:
            logger.error(f"Failed to ensure Qdrant collection '{self.collection_name}': {exc}")
            raise ValueError(
                f"Failed to ensure Qdrant collection '{self.collection_name}': {exc}"
            ) from exc

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks with dense and BM25 sparse vectors in Qdrant.

        Args:
            chunks: List of Chunk entities to upsert. Missing dense embeddings are computed via
                the configured embedder.

        Returns:
            None.

        Raises:
            ValueError: If embedding generation or Qdrant upsert fails.
        """
        if not chunks:
            return

        embeddings = await self._resolve_chunk_embeddings(chunks)
        await self.ensure_collection(vector_size=len(embeddings[0]))

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk.id)),
                vector={
                    _DENSE_VECTOR_NAME: embedding,
                    _SPARSE_VECTOR_NAME: models.Document(text=chunk.text, model=_BM25_MODEL),
                },
                payload=self._to_payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        try:
            await self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            logger.error(f"Failed to upsert {len(points)} chunks into Qdrant: {exc}")
            raise ValueError(f"Failed to upsert chunks into Qdrant: {exc}") from exc

    async def hybrid_search(self, query: Query, top_k: int = 10) -> list[Chunk]:
        """Perform hybrid dense + BM25 retrieval with Reciprocal Rank Fusion.

        Args:
            query: Query model containing text, filters, and optional dense embedding.
            top_k: Maximum number of fused chunks to return.

        Returns:
            Top-k retrieved chunks ordered by fused RRF relevance score.

        Raises:
            ValueError: If query embedding generation or Qdrant search fails.
        """
        if top_k <= 0:
            return []

        dense_embedding = await self._resolve_query_embedding(query)
        bm25_embedding = await self._embed_bm25_query(query.text)
        qdrant_filter = self._build_filter(query.filters)

        try:
            response = await self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_embedding,
                        using=_DENSE_VECTOR_NAME,
                        filter=qdrant_filter,
                        limit=top_k,
                    ),
                    models.Prefetch(
                        query=bm25_embedding,
                        using=_SPARSE_VECTOR_NAME,
                        filter=qdrant_filter,
                        limit=top_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.error(f"Hybrid search failed for collection '{self.collection_name}': {exc}")
            raise ValueError(f"Qdrant hybrid search failed: {exc}") from exc

        return [self._to_chunk(point) for point in response.points]

    async def _resolve_chunk_embeddings(self, chunks: list[Chunk]) -> list[list[float]]:
        """Return dense embeddings for all chunks, batch-embedding those without one.

        Args:
            chunks: Chunks to resolve embeddings for.

        Returns:
            Dense embedding vectors aligned with ``chunks``.

        Raises:
            ValueError: If the embedder fails or returns a mismatched vector count.
        """
        missing_indices = [idx for idx, chunk in enumerate(chunks) if chunk.embedding is None]
        computed: list[list[float]] = []

        if missing_indices:
            missing_texts = [chunks[idx].text for idx in missing_indices]
            try:
                computed = await self._embedder.embed_batch(missing_texts)
            except Exception as exc:
                logger.error(f"Failed to embed {len(missing_texts)} chunks: {exc}")
                raise ValueError(f"Failed to embed chunks: {exc}") from exc

            if len(computed) != len(missing_indices):
                raise ValueError(
                    f"Embedder returned {len(computed)} vectors for {len(missing_indices)} chunks"
                )

        computed_iter = iter(computed)
        return [
            chunk.embedding if chunk.embedding is not None else next(computed_iter)
            for chunk in chunks
        ]

    async def _resolve_query_embedding(self, query: Query) -> list[float]:
        """Return the query's dense embedding, computing it if absent.

        Args:
            query: Search query model.

        Returns:
            Dense vector representation of the query.

        Raises:
            ValueError: If embedding generation fails.
        """
        if query.embedding is not None:
            return query.embedding

        try:
            return await self._embedder.embed_text(query.text)
        except Exception as exc:
            logger.error(f"Failed to embed query '{query.text}': {exc}")
            raise ValueError(f"Failed to embed query: {exc}") from exc

    async def _embed_bm25_query(self, text: str) -> models.SparseVector:
        """Compute a BM25 sparse vector for a query string via FastEmbed.

        Args:
            text: Raw query text to tokenize and weight.

        Returns:
            SparseVector compatible with Qdrant's ``bm25`` index.

        Raises:
            ValueError: If FastEmbed model initialization or inference fails.
        """
        # qdrant-client 1.13 skips client-side Document inference when FusionQuery is paired with
        # a list of Prefetch objects, so we compute the sparse query vector explicitly.
        try:
            sparse_embedding = await asyncio.to_thread(self._compute_bm25_sparse, text)
        except Exception as exc:
            logger.error(f"Failed to compute BM25 sparse vector for query '{text}': {exc}")
            raise ValueError(f"Failed to compute BM25 query vector: {exc}") from exc
        return sparse_embedding

    def _compute_bm25_sparse(self, text: str) -> models.SparseVector:
        """Run synchronous FastEmbed BM25 inference to build a Qdrant SparseVector.

        Args:
            text: Query text to embed into sparse BM25 weights.

        Returns:
            SparseVector containing token indices and term weights.
        """
        if self._bm25_embedder is None:
            self._bm25_embedder = SparseTextEmbedding(model_name=_BM25_MODEL)
        embedding = next(iter(self._bm25_embedder.query_embed(text)))
        return models.SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> models.Filter | None:
        """Convert domain query filters into a Qdrant Filter.

        Args:
            filters: Key-value filter constraints from ``Query.filters``.

        Returns:
            A Qdrant ``Filter`` with ``must`` conditions, or ``None`` if empty.
        """
        if not filters:
            return None

        conditions: list[models.Condition] = []
        for raw_key, value in filters.items():
            key = (
                raw_key
                if raw_key in _TOP_LEVEL_PAYLOAD_KEYS or raw_key.startswith("metadata.")
                else f"metadata.{raw_key}"
            )
            if isinstance(value, list | tuple | set | frozenset):
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=list(value)))
                )
            else:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        return models.Filter(must=conditions)

    @staticmethod
    def _to_payload(chunk: Chunk) -> dict[str, Any]:
        """Serialize a Chunk into a Qdrant point payload.

        Args:
            chunk: Chunk entity to serialize.

        Returns:
            Dictionary payload stored alongside vectors in Qdrant.
        """
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }

    @staticmethod
    def _to_chunk(point: models.ScoredPoint) -> Chunk:
        """Map a Qdrant ScoredPoint back to a Chunk domain entity.

        Args:
            point: ScoredPoint returned by Qdrant ``query_points``.

        Returns:
            Reconstructed ``Chunk`` with preserved metadata and fused relevance score.

        Raises:
            ValueError: If required payload fields are missing from the point.
        """
        payload = point.payload or {}
        try:
            chunk_id = str(payload["chunk_id"])
            document_id = str(payload["document_id"])
            text = str(payload["text"])
        except KeyError as exc:
            raise ValueError(f"Qdrant point '{point.id}' is missing payload field {exc}") from exc

        raw_metadata = payload.get("metadata")
        metadata: dict[str, Any] = (
            dict(cast("dict[str, Any]", raw_metadata)) if isinstance(raw_metadata, dict) else {}
        )
        return Chunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            metadata=metadata,
            score=float(point.score),
        )
