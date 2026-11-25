"""Unit tests for ``POST /ask`` with a fake ``AgenticLoopService`` dependency."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentic_rag_copilot.api import APISettings, create_app
from agentic_rag_copilot.api._dependencies import get_agentic_loop_service
from agentic_rag_copilot.domain.exceptions import LLMError
from agentic_rag_copilot.domain.models import AskRequest, AskResponse, Citation
from agentic_rag_copilot.ingestion import IngestionSettings


class _FakeAgenticLoop:
    """Returns a configured ``AskResponse`` or raises a configured error, recording all requests."""

    def __init__(
        self,
        response: AskResponse | None = None,
        *,
        fail_with: Exception | None = None,
    ) -> None:
        """Initialize the fake loop.

        Args:
            response: Canned ``AskResponse`` to return on ``ask()``.
            fail_with: Optional exception to raise instead of returning ``response``.
        """
        self.response = response or AskResponse(
            query="What is RAG?",
            answer="RAG combines parametric and non-parametric memory [1].",
            citations=[
                Citation(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    source_uri="https://arxiv.org/pdf/2005.11401",
                    quote="non-parametric memory is a dense vector index of Wikipedia",
                )
            ],
            iterations_taken=1,
        )
        self.fail_with = fail_with
        self.requests: list[AskRequest] = []

    async def ask(self, request: AskRequest) -> AskResponse:
        """Record the incoming request and return the canned response or raise ``fail_with``.

        Args:
            request: Incoming ``AskRequest``.

        Returns:
            Configured ``AskResponse``.

        Raises:
            Exception: ``self.fail_with`` when configured.
        """
        self.requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        return self.response


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own ``config/.env`` from influencing the app under test.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary working directory.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fake_loop() -> _FakeAgenticLoop:
    """Provide a fresh fake agentic loop for each test.

    Returns:
        Configured ``_FakeAgenticLoop``.
    """
    return _FakeAgenticLoop()


@pytest.fixture
def client(fake_loop: _FakeAgenticLoop) -> Iterator[TestClient]:
    """Build a ``TestClient`` with ``get_agentic_loop_service`` overridden.

    Args:
        fake_loop: Fake loop injected into the FastAPI app.

    Yields:
        Active ``TestClient``.
    """
    app = create_app(APISettings(), IngestionSettings())
    app.dependency_overrides[get_agentic_loop_service] = lambda: fake_loop
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_ask_returns_200_with_exact_loop_response(
    client: TestClient, fake_loop: _FakeAgenticLoop
) -> None:
    """Verify ``POST /ask`` forwards ``AskRequest`` and returns ``AskResponse`` unchanged."""
    # Given
    payload = {
        "query": "What is RAG?",
        "top_k": 6,
        "filters": {"source_type": "pdf"},
    }

    # When
    response = client.post("/ask", json=payload)

    # Then
    assert response.status_code == 200
    assert response.json() == fake_loop.response.model_dump(mode="json")
    assert fake_loop.requests == [
        AskRequest(query="What is RAG?", top_k=6, filters={"source_type": "pdf"})
    ]


def test_ask_rejects_empty_query_with_422(client: TestClient, fake_loop: _FakeAgenticLoop) -> None:
    """Verify ``POST /ask`` returns 422 when ``query`` is empty."""
    # Given
    payload = {"query": ""}

    # When
    response = client.post("/ask", json=payload)

    # Then
    assert response.status_code == 422
    assert fake_loop.requests == []


@pytest.mark.parametrize("top_k", [0, 21])
def test_ask_rejects_out_of_range_top_k_with_422(
    client: TestClient, fake_loop: _FakeAgenticLoop, top_k: int
) -> None:
    """Verify ``POST /ask`` returns 422 when ``top_k`` is outside ``[1, 20]``."""
    # Given
    payload = {"query": "What is RAG?", "top_k": top_k}

    # When
    response = client.post("/ask", json=payload)

    # Then
    assert response.status_code == 422
    assert fake_loop.requests == []


def test_ask_maps_llm_error_to_503_without_leaking_details(
    client: TestClient, fake_loop: _FakeAgenticLoop
) -> None:
    """Verify ``LLMError`` maps to 503 with a generic JSON ``detail``."""
    # Given
    fake_loop.fail_with = LLMError("secret ollama stack detail")

    # When
    response = client.post("/ask", json={"query": "What is RAG?"})

    # Then
    assert response.status_code == 503
    assert set(response.json()) == {"detail"}
    assert "secret ollama stack detail" not in response.text


def test_ask_maps_synthesizer_value_error_to_502_without_leaking_details(
    client: TestClient, fake_loop: _FakeAgenticLoop
) -> None:
    """Verify synthesizer ``ValueError`` (unknown citation) maps to 502 without leaking details."""
    # Given
    fake_loop.fail_with = ValueError("LLM cited unknown chunk_id 'secret-hallucinated-id'")

    # When
    response = client.post("/ask", json={"query": "What is RAG?"})

    # Then
    assert response.status_code == 502
    assert set(response.json()) == {"detail"}
    assert "secret-hallucinated-id" not in response.text


def test_ask_openapi_schema_documents_route_and_error_responses(client: TestClient) -> None:
    """Verify FastAPI's generated OpenAPI schema documents ``POST /ask`` and its status codes."""
    # When
    response = client.get("/openapi.json")

    # Then
    assert response.status_code == 200
    post_ask = response.json()["paths"]["/ask"]["post"]
    assert post_ask["summary"] == "Ask a question over ingested documents"
    assert set(post_ask["responses"]) >= {"200", "422", "502", "503"}
