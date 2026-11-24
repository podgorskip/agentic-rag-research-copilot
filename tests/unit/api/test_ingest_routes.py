from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentic_rag_copilot.api import APISettings, create_app
from agentic_rag_copilot.api._dependencies import get_ingestion_pipeline
from agentic_rag_copilot.domain.exceptions import EmbeddingError
from agentic_rag_copilot.ingestion import IngestionError, IngestionSettings, IngestionStage
from tests.mocks.api import MockIngestionPipeline

_MAX_UPLOAD_BYTES = 1024
_PDF_BYTES = b"%PDF-1.4 minimal test payload"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own config/.env from influencing the app under test."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def pipeline() -> MockIngestionPipeline:
    return MockIngestionPipeline()


@pytest.fixture
def client(pipeline: MockIngestionPipeline) -> Iterator[TestClient]:
    app = create_app(
        APISettings(max_upload_bytes=_MAX_UPLOAD_BYTES),
        IngestionSettings(),
    )
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline
    with TestClient(app) as test_client:
        yield test_client


def test_json_source_is_ingested_and_returns_201(
    client: TestClient, pipeline: MockIngestionPipeline
) -> None:
    # When
    response = client.post(
        "/ingest",
        json={"source_uri": "https://arxiv.org/abs/2005.11401", "source_type": "html"},
    )

    # Then
    assert response.status_code == 201
    assert response.json() == {"document_id": "mock-document", "chunks_created": 1}
    assert [request.source_uri for request in pipeline.requests] == [
        "https://arxiv.org/abs/2005.11401"
    ]


def test_multipart_upload_is_ingested_from_a_stored_file(
    client: TestClient, pipeline: MockIngestionPipeline
) -> None:
    # When
    response = client.post(
        "/ingest",
        files={"file": ("paper.pdf", _PDF_BYTES, "application/pdf")},
    )

    # Then
    assert response.status_code == 201
    (request,) = pipeline.requests
    assert request.source_uri.endswith(".pdf")
    assert request.source_type.value == "pdf"


def test_uploaded_file_is_deleted_after_ingestion(
    client: TestClient, pipeline: MockIngestionPipeline
) -> None:
    # When
    client.post("/ingest", files={"file": ("paper.pdf", _PDF_BYTES, "application/pdf")})

    # Then
    (request,) = pipeline.requests
    assert not Path(request.source_uri).exists()


def test_identical_uploads_share_a_source_path(
    client: TestClient, pipeline: MockIngestionPipeline
) -> None:
    # When
    for _ in range(2):
        client.post("/ingest", files={"file": ("paper.pdf", _PDF_BYTES, "application/pdf")})

    # Then
    first, second = pipeline.requests
    assert first.source_uri == second.source_uri


def test_unsupported_json_source_type_returns_422(
    client: TestClient, pipeline: MockIngestionPipeline
) -> None:
    # When
    response = client.post(
        "/ingest",
        json={"source_uri": "notes.docx", "source_type": "docx"},
    )

    # Then
    assert response.status_code == 422
    assert pipeline.requests == []


def test_upload_with_unsupported_extension_returns_422(
    client: TestClient, pipeline: MockIngestionPipeline
) -> None:
    # When
    response = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    # Then
    assert response.status_code == 422
    assert pipeline.requests == []


def test_multipart_without_file_field_returns_422(client: TestClient) -> None:
    # When
    response = client.post("/ingest", files={"other": ("paper.pdf", _PDF_BYTES)})

    # Then
    assert response.status_code == 422
    assert "'file' field" in response.json()["detail"]


def test_oversized_upload_returns_413_before_ingestion(
    client: TestClient, pipeline: MockIngestionPipeline
) -> None:
    # Given
    oversized = b"%PDF" + b"x" * (_MAX_UPLOAD_BYTES + 1)

    # When
    response = client.post("/ingest", files={"file": ("big.pdf", oversized, "application/pdf")})

    # Then
    assert response.status_code == 413
    assert pipeline.requests == []


def test_pipeline_ingestion_failure_returns_422_without_source_path(
    client: TestClient, pipeline: MockIngestionPipeline
) -> None:
    # Given
    pipeline.fail_with = IngestionError(
        "/secret/path/paper.pdf", IngestionStage.LOAD, "corrupt header at /secret/path"
    )

    # When
    response = client.post(
        "/ingest",
        json={"source_uri": "/secret/path/paper.pdf", "source_type": "pdf"},
    )

    # Then
    assert response.status_code == 422
    assert response.json() == {"detail": "Ingestion failed during the load stage"}


def test_infrastructure_failure_returns_503(
    client: TestClient, pipeline: MockIngestionPipeline
) -> None:
    # Given
    pipeline.fail_with = EmbeddingError("ollama down")

    # When
    response = client.post(
        "/ingest",
        json={"source_uri": "https://arxiv.org/abs/2005.11401", "source_type": "html"},
    )

    # Then
    assert response.status_code == 503
    assert set(response.json()) == {"detail"}
