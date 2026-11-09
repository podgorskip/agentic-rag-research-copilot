import hashlib
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag_copilot.domain.models import Chunk, Document
from agentic_rag_copilot.domain.protocols import ChunkerProtocol


class LangchainTextChunker(ChunkerProtocol):
    """Intelligently chunks text while respecting paragraphs and sentences."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initialize the text chunker with splitting parameters.

        Args:
            chunk_size: Maximum size of each text chunk in characters (default: 1000).
            chunk_overlap: Number of characters to overlap
                           between consecutive chunks (default: 200).
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Synchronously splits document text into semantic chunks.

        Args:
            document: The document to chunk.

        Returns:
            A list of Chunk objects with unique IDs, text, document reference, and metadata.
        """
        text = document.raw_content
        if not text:
            return []

        split_texts = self.splitter.split_text(text)

        chunks: list[Chunk] = []
        for index, chunk_text in enumerate(split_texts):
            hash_input = f"{document.id}_{index}".encode()
            chunk_id = hashlib.sha256(hash_input).hexdigest()
            metadata: dict[str, Any] = document.metadata.copy()
            metadata["chunk_index"] = index

            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

        return chunks
