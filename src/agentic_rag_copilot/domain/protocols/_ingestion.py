from typing import Protocol, runtime_checkable

from ..models import Chunk, Document


@runtime_checkable
class DocumentLoaderProtocol(Protocol):
    """Contract for PyMuPDF / trafilatura adapters."""

    async def load(self, source_uri: str) -> Document:
        """Load document content from a source URI.

        Args:
            source_uri: URL or local file path to the document.

        Returns:
            Extracted Document entity.
        """
        ...


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Contract for text splitting adapters."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into indexable chunks.

        Synchronous because chunking is purely an in-memory string operation.

        Args:
            document: Document entity to be split into chunks.

        Returns:
            List of generated text chunks.
        """
        ...
