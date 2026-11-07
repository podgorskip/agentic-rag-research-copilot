import asyncio
import hashlib
import urllib.request
from typing import Any

import trafilatura

from agentic_rag_copilot.domain.models import Document, SourceType
from agentic_rag_copilot.domain.protocols import DocumentLoaderProtocol


class TrafilaturaHTMLLoader(DocumentLoaderProtocol):
    """Adapter for fetching and extracting clean text from URLs."""

    async def load(self, source_uri: str) -> Document:
        """Asynchronously fetches a URL and extracts its main body text."""
        return await asyncio.to_thread(self._sync_load, source_uri)

    @staticmethod
    def _sync_load(source_uri: str) -> Document:
        """Synchronous implementation isolated to a worker thread."""

        try:
            req = urllib.request.Request(
                source_uri, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                downloaded_html = response.read()
        except Exception as e:
            raise ValueError(f"Failed to fetch HTML from: {source_uri}. Reason: {e!s}") from e

        try:
            text = trafilatura.extract(downloaded_html, include_links=True)
        except Exception as e:
            raise ValueError(
                f"Failed to extract meaningful text from: {source_uri}. Parser error: {e!s}"
            ) from e

        if not text or not str(text).strip():
            raise ValueError(f"Failed to extract meaningful text from: {source_uri}")

        doc_id = hashlib.sha256(source_uri.encode("utf-8")).hexdigest()

        metadata: dict[str, Any] = {
            "parser": "trafilatura",
        }

        return Document(
            id=doc_id,
            source_type=SourceType.HTML,
            source_uri=source_uri,
            raw_content=str(text).strip(),
            metadata=metadata,
        )
