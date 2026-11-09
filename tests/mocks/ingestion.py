"""Deterministic in-memory ingestion adapters, for use in the test suite only.

Each mock records how it was called and can be told to fail, so a test can exercise both the
happy path and error handling without PDFs, network calls or ML models.
"""

import hashlib
import math
from collections.abc import Mapping

from agentic_rag_copilot.domain.models import Chunk, Document, Query, SourceType
from agentic_rag_copilot.domain.protocols import (
    ChunkerProtocol,
    DocumentLoaderProtocol,
    EmbedderProtocol,
    VectorStoreProtocol,
)


class MockDocumentLoader(DocumentLoaderProtocol):
    """Returns configured documents, or synthesises a stable one for any other URI."""

    def __init__(
        self,
        documents: Mapping[str, Document] | None = None,
        *,
        failures: Mapping[str, Exception] | None = None,
        source_type: SourceType = SourceType.PDF,
        content: str = "Mock document content used to exercise the ingestion flow.",
    ) -> None:
        self.documents = dict(documents or {})
        self.failures = dict(failures or {})
        self.source_type = source_type
        self.content = content
        self.loaded_uris: list[str] = []

    async def load(self, source_uri: str) -> Document:
        self.loaded_uris.append(source_uri)
        if source_uri in self.failures:
            raise self.failures[source_uri]
        if source_uri in self.documents:
            return self.documents[source_uri]
        return Document(
            id=hashlib.sha256(source_uri.encode()).hexdigest(),
            source_type=self.source_type,
            source_uri=source_uri,
            raw_content=self.content,
        )


class MockChunker(ChunkerProtocol):
    """Splits text into fixed-size windows, keeping the metadata contract of the real chunker."""

    def __init__(self, chunk_size: int = 20, *, fail_with: Exception | None = None) -> None:
        self.chunk_size = chunk_size
        self.fail_with = fail_with
        self.chunked_documents: list[Document] = []

    def chunk(self, document: Document) -> list[Chunk]:
        self.chunked_documents.append(document)
        if self.fail_with is not None:
            raise self.fail_with
        text = document.raw_content
        return [
            Chunk(
                id=f"{document.id}-{index}",
                document_id=document.id,
                text=text[start : start + self.chunk_size],
                metadata={**document.metadata, "chunk_index": index},
            )
            for index, start in enumerate(range(0, len(text), self.chunk_size))
        ]


class MockEmbedder(EmbedderProtocol):
    """Derives a unit vector from a hash of the text: equal texts get equal vectors."""

    def __init__(self, dimension: int = 8, *, fail_with: Exception | None = None) -> None:
        self.dimension = dimension
        self.fail_with = fail_with
        self.batches: list[list[str]] = []

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self.fail_with is not None:
            raise self.fail_with
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.shake_256(text.encode()).digest(self.dimension)
        # Centring on 127.5 means no component can be zero, so the norm is never zero.
        raw = [(byte - 127.5) / 127.5 for byte in digest]
        norm = math.hypot(*raw)
        return [value / norm for value in raw]


class InMemoryVectorStore(VectorStoreProtocol):
    """Keeps upserted chunks in a dict and, like Qdrant, rejects chunks without embeddings."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.chunks: dict[str, Chunk] = {}

    async def upsert(self, chunks: list[Chunk]) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk '{chunk.id}' has no embedding and cannot be upserted")
            self.chunks[chunk.id] = chunk

    async def hybrid_search(self, query: Query, top_k: int = 10) -> list[Chunk]:
        return list(self.chunks.values())[:top_k]
