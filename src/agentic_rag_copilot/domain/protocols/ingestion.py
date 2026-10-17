from typing import Protocol, runtime_checkable

from ..models.entities import Chunk, Document


@runtime_checkable
class DocumentLoaderProtocol(Protocol):
    """Contract for PyMuPDF / trafilatura adapters."""

    async def load(self, source_uri: str) -> Document: ...


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Contract for text splitting adapters."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Synchronous because chunking is purely an in-memory string operation."""
        ...
