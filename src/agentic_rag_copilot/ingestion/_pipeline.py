from collections.abc import Generator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from agentic_rag_copilot.domain.models import IngestRequest, IngestResponse, SourceType
from agentic_rag_copilot.domain.protocols import (
    ChunkerProtocol,
    DocumentLoaderProtocol,
    EmbedderProtocol,
    VectorStoreProtocol,
)

_ARXIV_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})


class IngestionStage(StrEnum):
    LOAD = "load"
    CHUNK = "chunk"
    EMBED = "embed"
    UPSERT = "upsert"


class IngestionError(Exception):
    """A source could not be ingested; names the source and the stage that failed."""

    def __init__(self, source_uri: str, stage: IngestionStage, reason: str) -> None:
        super().__init__(f"{stage} stage failed: {reason}")
        self.source_uri = source_uri
        self.stage = stage


@contextmanager
def _failing_as(stage: IngestionStage, source_uri: str) -> Generator[None]:
    # Adapters surface many unrelated exception types (OSError, ValueError, HTTP and Qdrant
    # errors), so this boundary deliberately normalises all of them.
    try:
        yield
    except Exception as exc:
        raise IngestionError(source_uri, stage, str(exc) or repr(exc)) from exc


def detect_source_type(source_uri: str) -> SourceType:
    """Infer whether a source is a PDF or an HTML page from its URI.

    Args:
        source_uri: URL or local file path.

    Returns:
        The detected source type.

    Raises:
        ValueError: If the URI matches neither supported type.
    """
    parts = urlsplit(source_uri)
    # A bare file path may contain "?" or "#", which URL parsing would treat as delimiters.
    path = parts.path if parts.scheme else source_uri
    suffix = PurePosixPath(path).suffix.lower()

    if suffix == ".pdf":
        return SourceType.PDF
    if suffix in {".html", ".htm"}:
        return SourceType.HTML
    if parts.scheme in {"http", "https"}:
        is_arxiv_pdf = parts.hostname in _ARXIV_HOSTS and parts.path.startswith("/pdf/")
        return SourceType.PDF if is_arxiv_pdf else SourceType.HTML
    raise ValueError(f"Cannot determine source type of '{source_uri}'; expected a PDF or a URL")


class IngestionPipeline:
    """Runs the Load -> Chunk -> Embed -> Upsert flow against the domain protocols."""

    def __init__(
        self,
        loaders: Mapping[SourceType, DocumentLoaderProtocol],
        chunker: ChunkerProtocol,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
    ) -> None:
        self._loaders = loaders
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        """Ingest one source into the vector store.

        Args:
            request: Source URI and its type.

        Returns:
            The ID of the ingested document and the number of chunks stored.

        Raises:
            IngestionError: If any stage fails, including when a stage yields nothing to store.
        """
        uri = request.source_uri

        with _failing_as(IngestionStage.LOAD, uri):
            document = await self._loaders[request.source_type].load(uri)

        with _failing_as(IngestionStage.CHUNK, uri):
            chunks = self._chunker.chunk(document)
            if not chunks:
                raise ValueError("chunker produced no chunks")

        with _failing_as(IngestionStage.EMBED, uri):
            vectors = await self._embedder.embed_batch([chunk.text for chunk in chunks])
            embedded = [
                chunk.model_copy(update={"embedding": vector})
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]

        with _failing_as(IngestionStage.UPSERT, uri):
            await self._vector_store.upsert(embedded)

        return IngestResponse(document_id=document.id, chunks_created=len(embedded))
