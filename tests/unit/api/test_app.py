from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_rag_copilot.api import APISettings, create_app
from agentic_rag_copilot.api._dependencies import get_probe_client
from agentic_rag_copilot.domain.exceptions import EmbeddingError, LLMError
from agentic_rag_copilot.ingestion import IngestionSettings


def _probe(reachable: set[str]) -> httpx.AsyncClient:
    """Probe client whose upstreams answer only for the hosts listed in ``reachable``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host in reachable:
            return httpx.Response(200, json={})
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own config/.env from influencing the app under test."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def make_client() -> Iterator[Callable[..., TestClient]]:
    """Build a test client for an app whose upstream probes are faked."""
    clients: list[TestClient] = []

    def build(
        reachable: set[str] | None = None,
        *,
        cors_origins: list[str] | None = None,
    ) -> TestClient:
        app = create_app(
            APISettings(cors_origins=cors_origins or ["http://ui.test"]),
            IngestionSettings(),
        )
        probe = _probe(reachable or set())
        app.dependency_overrides[get_probe_client] = lambda: probe
        client = TestClient(app)
        clients.append(client)
        return client.__enter__()

    yield build
    for client in clients:
        client.__exit__(None, None, None)


def test_create_app_returns_health_route() -> None:
    # When
    app = create_app(APISettings(), IngestionSettings())

    # Then
    assert isinstance(app, FastAPI)
    assert any(getattr(route, "path", None) == "/health" for route in app.routes)


def test_health_reports_all_upstreams_reachable(make_client: Callable[..., TestClient]) -> None:
    # Given
    client = make_client(reachable={"localhost"})

    # When
    response = client.get("/health")

    # Then
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant": True, "ollama": True}


def test_health_reports_unreachable_upstream_as_false(
    make_client: Callable[..., TestClient],
) -> None:
    # Given
    app_client = make_client(reachable=set())

    # When
    response = app_client.get("/health")

    # Then
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant": False, "ollama": False}


@pytest.mark.parametrize("error", [LLMError("ollama exploded: secret detail"), EmbeddingError("x")])
def test_infrastructure_errors_map_to_503_without_leaking_details(
    make_client: Callable[..., TestClient], error: Exception
) -> None:
    # Given
    client = make_client()
    app = client.app
    assert isinstance(app, FastAPI)

    @app.get("/_boom")
    async def boom() -> None:
        raise error

    client = TestClient(app, raise_server_exceptions=False)

    # When
    response = client.get("/_boom")

    # Then
    assert response.status_code == 503
    assert set(response.json()) == {"detail"}
    assert "secret detail" not in response.text


def test_cors_allows_configured_origin(make_client: Callable[..., TestClient]) -> None:
    # Given
    client = make_client(cors_origins=["http://ui.test"])

    # When
    response = client.get("/health", headers={"Origin": "http://ui.test"})

    # Then
    assert response.headers["access-control-allow-origin"] == "http://ui.test"
