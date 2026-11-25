"""Unit tests for ``POST /ask/stream`` SSE framing and lifecycle handling."""

import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentic_rag_copilot.api import APISettings, create_app
from agentic_rag_copilot.api._dependencies import get_agentic_loop_service
from agentic_rag_copilot.domain.models import AgentEvent, AskRequest, AskResponse, Citation
from agentic_rag_copilot.ingestion import IngestionSettings


class _FakeStreamingLoop:
    """Yields a fixed list of ``AgentEvent`` objects and tracks generator closure."""

    def __init__(self, events: list[AgentEvent]) -> None:
        self._events = list(events)
        self.requests: list[AskRequest] = []
        self.closed = False

    async def ask_stream(self, request: AskRequest) -> AsyncGenerator[AgentEvent]:
        self.requests.append(request)
        try:
            for event in self._events:
                yield event
        finally:
            self.closed = True


def _parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    """Parse SSE text into ``(event_name, json_data)`` tuples.

    Args:
        body: Raw ``text/event-stream`` response body.

    Returns:
        List of parsed ``(event_name, payload_dict)`` tuples.
    """
    frames: list[tuple[str, dict[str, object]]] = []
    for raw_frame in body.strip().split("\n\n"):
        if not raw_frame.strip():
            continue
        lines = raw_frame.splitlines()
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        event_name = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        frames.append((event_name, data))
    return frames


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own ``config/.env`` from influencing the app under test.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary working directory.
    """
    monkeypatch.chdir(tmp_path)


def _make_client(fake_loop: _FakeStreamingLoop) -> TestClient:
    """Build a ``TestClient`` with ``get_agentic_loop_service`` overridden.

    Args:
        fake_loop: Streaming loop fake.

    Returns:
        Configured ``TestClient``.
    """
    app = create_app(APISettings(), IngestionSettings())
    app.dependency_overrides[get_agentic_loop_service] = lambda: fake_loop
    return TestClient(app, raise_server_exceptions=False)


def test_ask_stream_emits_sse_frames_headers_and_terminal_done() -> None:
    """Verify ``POST /ask/stream`` returns SSE headers, valid framing, and ``done`` last."""
    # Given
    answer = AskResponse(
        query="What is RAG?",
        answer="RAG combines parametric and non-parametric memory [1].",
        citations=[
            Citation(
                chunk_id="c1",
                document_id="doc-1",
                source_uri="https://arxiv.org/pdf/2005.11401",
                quote="non-parametric memory is a dense vector index",
            )
        ],
        iterations_taken=1,
    )
    fake_loop = _FakeStreamingLoop(
        [
            AgentEvent(type="retrieve", iteration=1, payload={"query": "What is RAG?"}),
            AgentEvent(
                type="grade",
                iteration=1,
                payload={"is_context_sufficient": True, "reasoning": "sufficient"},
            ),
            AgentEvent(type="generate", iteration=1, payload={"query": "What is RAG?"}),
            AgentEvent(type="answer", iteration=1, payload=answer.model_dump(mode="json")),
        ]
    )

    # When
    with _make_client(fake_loop) as client:
        response = client.post("/ask/stream", json={"query": "What is RAG?", "top_k": 4})

    # Then
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    frames = _parse_sse(response.text)
    assert [name for name, _ in frames] == ["retrieve", "grade", "generate", "answer", "done"]
    assert AgentEvent.model_validate(frames[3][1]).payload == answer.model_dump(mode="json")
    assert frames[-1] == ("done", {})
    assert fake_loop.requests == [AskRequest(query="What is RAG?", top_k=4)]
    assert fake_loop.closed is True


def test_ask_stream_forwards_mid_stream_error_and_closes_without_done() -> None:
    """Verify a mid-stream ``error`` event is forwarded and closes the stream immediately."""
    # Given
    fake_loop = _FakeStreamingLoop(
        [
            AgentEvent(type="retrieve", iteration=1, payload={"query": "What is RAG?"}),
            AgentEvent(
                type="error",
                iteration=1,
                payload={"detail": "An error occurred", "error_type": "LLMError"},
            ),
            AgentEvent(type="answer", iteration=1, payload={}),
        ]
    )

    # When
    with _make_client(fake_loop) as client:
        response = client.post("/ask/stream", json={"query": "What is RAG?"})

    # Then
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(response.text)
    assert [name for name, _ in frames] == ["retrieve", "error"]
    assert frames[1][1]["payload"] == {"detail": "An error occurred", "error_type": "LLMError"}
    assert fake_loop.closed is True


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "q", "top_k": 0}])
def test_ask_stream_rejects_invalid_request_with_422(payload: dict[str, object]) -> None:
    """Verify ``POST /ask/stream`` validates ``AskRequest`` before starting the stream."""
    # Given
    fake_loop = _FakeStreamingLoop([])

    # When
    with _make_client(fake_loop) as client:
        response = client.post("/ask/stream", json=payload)

    # Then
    assert response.status_code == 422
    assert fake_loop.requests == []


def test_ask_stream_openapi_schema_documents_event_stream() -> None:
    """Verify OpenAPI schema documents ``POST /ask/stream`` with ``text/event-stream``."""
    # Given
    fake_loop = _FakeStreamingLoop([])

    # When
    with _make_client(fake_loop) as client:
        response = client.get("/openapi.json")

    # Then
    assert response.status_code == 200
    post_stream = response.json()["paths"]["/ask/stream"]["post"]
    assert post_stream["summary"] == "Stream agentic RAG progress and answer over SSE"
    assert "text/event-stream" in post_stream["responses"]["200"]["content"]
    assert "422" in post_stream["responses"]
