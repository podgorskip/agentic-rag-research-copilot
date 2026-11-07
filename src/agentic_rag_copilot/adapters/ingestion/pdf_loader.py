import asyncio
import hashlib
from pathlib import Path
from typing import Any, cast

import pymupdf

from agentic_rag_copilot.domain.models import Document, SourceType
from agentic_rag_copilot.domain.protocols.ingestion import DocumentLoaderProtocol


class PyMuPDFLoader(DocumentLoaderProtocol):
    """Adapter for loading and extracting text from PDF files using PyMuPDF."""

    async def load(self, source_uri: str) -> Document:
        """Asynchronously loads a PDF file and extracts its text."""
        return await asyncio.to_thread(self._sync_load, source_uri)

    @staticmethod
    def _sync_load(source_uri: str) -> Document:
        """Synchronous implementation isolated to a worker thread."""
        path = Path(source_uri)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {source_uri}")

        try:
            with pymupdf.open(str(path)) as doc:
                text_pages: list[str] = []
                for page in doc:
                    text_pages.append(str(page.get_text("text")))

                raw_content = "\n".join(text_pages).strip()

                if not raw_content:
                    raise ValueError(f"No extractable text found in PDF: {source_uri}")

                pdf_meta = cast("dict[str, Any]", doc.metadata or {})

                metadata: dict[str, Any] = {
                    "page_count": doc.page_count,
                    "author": pdf_meta.get("author", ""),
                    "title": pdf_meta.get("title", ""),
                }

        except Exception as e:
            raise ValueError(f"Failed to parse PDF {source_uri}: {e!s}") from e

        doc_id = hashlib.sha256(source_uri.encode("utf-8")).hexdigest()

        return Document(
            id=doc_id,
            source_type=SourceType.PDF,
            source_uri=source_uri,
            raw_content=raw_content,
            metadata=metadata,
        )
