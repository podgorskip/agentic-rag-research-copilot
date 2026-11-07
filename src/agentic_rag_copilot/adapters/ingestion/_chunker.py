import hashlib
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag_copilot.domain.models import Chunk, Document
from agentic_rag_copilot.domain.protocols import ChunkerProtocol


class LangchainTextChunker(ChunkerProtocol):
    """Intelligently chunks text while respecting paragraphs and sentences."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        # We delegate the complex validation and splitting logic to LangChain
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Synchronously splits document text into semantic chunks."""
        text = document.raw_content
        if not text:
            return []

        # Langchain handles the intelligent text splitting
        split_texts = self.splitter.split_text(text)

        chunks: list[Chunk] = []
        for index, chunk_text in enumerate(split_texts):
            # SHA-256 for SonarCloud compliance
            hash_input = f"{document.id}_{index}".encode()
            chunk_id = hashlib.sha256(hash_input).hexdigest()

            # Inherit parent metadata and append chunk-specific data
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
