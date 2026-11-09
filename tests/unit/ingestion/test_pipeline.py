import pytest

from agentic_rag_copilot.domain.models import Chunk, Document, IngestRequest, SourceType
from agentic_rag_copilot.ingestion import (
    IngestionError,
    IngestionPipeline,
    IngestionStage,
    detect_source_type,
)
from tests.mocks.ingestion import (
    InMemoryVectorStore,
    MockChunker,
    MockDocumentLoader,
    MockEmbedder,
)

PDF_REQUEST = IngestRequest(source_uri="paper.pdf", source_type=SourceType.PDF)


def build_pipeline(
    loader: MockDocumentLoader | None = None,
    chunker: MockChunker | None = None,
    embedder: MockEmbedder | None = None,
    store: InMemoryVectorStore | None = None,
) -> IngestionPipeline:
    return IngestionPipeline(
        loaders={SourceType.PDF: loader or MockDocumentLoader()},
        chunker=chunker or MockChunker(),
        embedder=embedder or MockEmbedder(),
        vector_store=store or InMemoryVectorStore(),
    )


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("paper.pdf", SourceType.PDF),
        ("/data/PAPER.PDF", SourceType.PDF),
        ("my#1.pdf", SourceType.PDF),
        ("what?.pdf", SourceType.PDF),
        ("https://example.org/paper.pdf?download=1", SourceType.PDF),
        ("https://arxiv.org/pdf/2401.00001v2", SourceType.PDF),
        ("https://arxiv.org/abs/2401.00001", SourceType.HTML),
        ("https://docs.example.org/guide", SourceType.HTML),
        ("https://docs.example.org/pdf/guide", SourceType.HTML),
        ("file:///tmp/page.htm", SourceType.HTML),
    ],
)
def test_detect_source_type(uri: str, expected: SourceType) -> None:
    assert detect_source_type(uri) is expected


@pytest.mark.parametrize("uri", ["notes.txt", "folder/file", "ftp://host/data"])
def test_detect_source_type_rejects_unknown(uri: str) -> None:
    with pytest.raises(ValueError, match="Cannot determine source type"):
        detect_source_type(uri)


async def test_ingest_stores_embedded_chunks() -> None:
    embedder = MockEmbedder()
    store = InMemoryVectorStore()
    loader = MockDocumentLoader(content="x" * 50)

    response = await build_pipeline(loader, MockChunker(chunk_size=20), embedder, store).ingest(
        PDF_REQUEST
    )

    assert response.chunks_created == 3
    assert len(embedder.batches) == 1
    assert len(store.chunks) == 3
    for chunk in store.chunks.values():
        assert chunk.document_id == response.document_id
        assert chunk.embedding == await embedder.embed_text(chunk.text)


async def test_ingest_routes_by_source_type() -> None:
    pdf_loader = MockDocumentLoader()
    html_loader = MockDocumentLoader(source_type=SourceType.HTML)
    pipeline = IngestionPipeline(
        loaders={SourceType.PDF: pdf_loader, SourceType.HTML: html_loader},
        chunker=MockChunker(),
        embedder=MockEmbedder(),
        vector_store=InMemoryVectorStore(),
    )

    await pipeline.ingest(IngestRequest(source_uri="https://x.test", source_type=SourceType.HTML))

    assert html_loader.loaded_uris == ["https://x.test"]
    assert pdf_loader.loaded_uris == []


async def test_load_failure_stops_the_flow() -> None:
    loader = MockDocumentLoader(failures={"paper.pdf": FileNotFoundError("no such file")})
    chunker = MockChunker()

    with pytest.raises(IngestionError, match="no such file") as caught:
        await build_pipeline(loader=loader, chunker=chunker).ingest(PDF_REQUEST)

    assert caught.value.stage is IngestionStage.LOAD
    assert caught.value.source_uri == "paper.pdf"
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert chunker.chunked_documents == []


async def test_missing_loader_is_a_load_failure() -> None:
    request = IngestRequest(source_uri="https://x.test", source_type=SourceType.HTML)

    with pytest.raises(IngestionError) as caught:
        await build_pipeline().ingest(request)

    assert caught.value.stage is IngestionStage.LOAD


async def test_chunk_failure_is_reported() -> None:
    chunker = MockChunker(fail_with=RuntimeError("bad split"))

    with pytest.raises(IngestionError, match="chunk stage failed: bad split"):
        await build_pipeline(chunker=chunker).ingest(PDF_REQUEST)


async def test_no_chunks_is_a_failure_and_nothing_is_embedded() -> None:
    class EmptyChunker(MockChunker):
        def chunk(self, document: Document) -> list[Chunk]:
            return []

    embedder = MockEmbedder()

    with pytest.raises(IngestionError, match="no chunks") as caught:
        await build_pipeline(chunker=EmptyChunker(), embedder=embedder).ingest(PDF_REQUEST)

    assert caught.value.stage is IngestionStage.CHUNK
    assert embedder.batches == []


async def test_embed_failure_stores_nothing() -> None:
    store = InMemoryVectorStore()
    embedder = MockEmbedder(fail_with=ConnectionError("refused"))

    with pytest.raises(IngestionError, match="embed stage failed: refused"):
        await build_pipeline(embedder=embedder, store=store).ingest(PDF_REQUEST)

    assert store.chunks == {}


async def test_vector_count_mismatch_stores_nothing() -> None:
    class ShortEmbedder(MockEmbedder):
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return []

    store = InMemoryVectorStore()

    with pytest.raises(IngestionError) as caught:
        await build_pipeline(embedder=ShortEmbedder(), store=store).ingest(PDF_REQUEST)

    assert caught.value.stage is IngestionStage.EMBED
    assert store.chunks == {}


async def test_upsert_failure_is_reported() -> None:
    store = InMemoryVectorStore(fail_with=ConnectionError("qdrant down"))

    with pytest.raises(IngestionError, match="upsert stage failed: qdrant down"):
        await build_pipeline(store=store).ingest(PDF_REQUEST)


async def test_exception_without_message_still_has_a_reason() -> None:
    embedder = MockEmbedder(fail_with=TimeoutError())

    with pytest.raises(IngestionError, match="TimeoutError"):
        await build_pipeline(embedder=embedder).ingest(PDF_REQUEST)
