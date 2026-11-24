"""``POST /ingest`` integration tests with the real PDF loader, chunker and Qdrant store.

Qdrant runs in-memory and embeddings come from ``MockEmbedder``, so the tests need neither
Docker nor a running Ollama instance.
"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import httpx
import pytest
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.ingestion import (
    LangchainTextChunker,
    PyMuPDFLoader,
    TrafilaturaHTMLLoader,
)
from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.api import APISettings, create_app
from agentic_rag_copilot.api._dependencies import get_api_settings, get_ingestion_pipeline
from agentic_rag_copilot.domain.models import SourceType
from agentic_rag_copilot.ingestion import IngestionPipeline, IngestionSettings
from tests.mocks.ingestion import MockEmbedder

PAPER_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "e2e" / "data" / "paper-01.pdf"
COLLECTION: Final[str] = "integration_ingest_endpoint"
UPLOAD_LIMIT: Final[int] = 10 * 1024 * 1024
SMALL_UPLOAD_LIMIT: Final[int] = 1024


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own config/.env from influencing the app under test."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


@pytest.fixture
def pipeline(qdrant: AsyncQdrantClient) -> IngestionPipeline:
    settings = IngestionSettings()
    embedder = MockEmbedder()
    return IngestionPipeline(
        loaders={
            SourceType.PDF: PyMuPDFLoader(),
            SourceType.HTML: TrafilaturaHTMLLoader(),
        },
        chunker=LangchainTextChunker(settings.chunk_size, settings.chunk_overlap),
        embedder=embedder,
        vector_store=QdrantVectorStore(qdrant, embedder, COLLECTION),
    )


@pytest.fixture
def app(pipeline: IngestionPipeline) -> FastAPI:
    api_settings = APISettings(max_upload_bytes=UPLOAD_LIMIT)
    app = create_app(api_settings, IngestionSettings())
    app.dependency_overrides[get_api_settings] = lambda: api_settings
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline
    return app


@pytest.fixture
async def http(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _point_count(qdrant: AsyncQdrantClient) -> int:
    if not await qdrant.collection_exists(COLLECTION):
        return 0
    return (await qdrant.count(COLLECTION)).count


async def test_pdf_upload_is_chunked_and_stored(
    http: httpx.AsyncClient, qdrant: AsyncQdrantClient
) -> None:
    # When
    response = await http.post(
        "/ingest",
        files={"file": ("paper-01.pdf", PAPER_PATH.read_bytes(), "application/pdf")},
    )

    # Then
    assert response.status_code == 201
    body = response.json()
    assert body["chunks_created"] > 0
    assert await _point_count(qdrant) == body["chunks_created"]


async def test_reingesting_the_same_pdf_is_idempotent(
    http: httpx.AsyncClient, qdrant: AsyncQdrantClient
) -> None:
    # Given
    payload = PAPER_PATH.read_bytes()
    first = await http.post("/ingest", files={"file": ("paper-01.pdf", payload, "application/pdf")})
    count_after_first = await _point_count(qdrant)

    # When
    second = await http.post(
        "/ingest", files={"file": ("paper-01.pdf", payload, "application/pdf")}
    )

    # Then
    assert second.status_code == 201
    assert second.json()["document_id"] == first.json()["document_id"]
    assert await _point_count(qdrant) == count_after_first


async def test_json_request_for_a_local_pdf_is_ingested(
    http: httpx.AsyncClient, qdrant: AsyncQdrantClient
) -> None:
    # When
    response = await http.post(
        "/ingest",
        json={"source_uri": str(PAPER_PATH), "source_type": "pdf"},
    )

    # Then
    assert response.status_code == 201
    assert response.json()["chunks_created"] > 0
    assert await _point_count(qdrant) > 0


async def test_oversized_upload_is_rejected_before_qdrant_is_touched(
    app: FastAPI, http: httpx.AsyncClient, qdrant: AsyncQdrantClient
) -> None:
    # Given
    payload = PAPER_PATH.read_bytes()
    small_limit = APISettings(max_upload_bytes=SMALL_UPLOAD_LIMIT)
    app.dependency_overrides[get_api_settings] = lambda: small_limit
    assert len(payload) > SMALL_UPLOAD_LIMIT

    # When
    response = await http.post(
        "/ingest", files={"file": ("paper-01.pdf", payload, "application/pdf")}
    )

    # Then
    assert response.status_code == 413
    assert not await qdrant.collection_exists(COLLECTION)
