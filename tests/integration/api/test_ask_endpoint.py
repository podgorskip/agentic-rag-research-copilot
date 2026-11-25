"""``POST /ask`` integration tests exercising the real ``create_app`` lifespan and service wiring.

Qdrant runs in-memory with ``MockEmbedder`` and ``ScriptedLoopLLM`` injected at the leaf adapter
boundary, so the real ``IngestionPipeline``, ``RetrievalService``, and ``AgenticLoopService``
wiring runs deterministically without Docker or Ollama.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker
from agentic_rag_copilot.agent import AgentSettings
from agentic_rag_copilot.api import APISettings, create_app
from agentic_rag_copilot.domain.exceptions import LLMError
from agentic_rag_copilot.ingestion import IngestionSettings
from tests.integration.conftest import SAMPLE_PAPER_PATH, stored_chunk_ids
from tests.mocks.agent import ScriptedLoopLLM
from tests.mocks.ingestion import MockEmbedder

_COLLECTION: Final[str] = "integration_ask_endpoint"
_UPLOAD_LIMIT: Final[int] = 10 * 1024 * 1024


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own ``config/.env`` from influencing the app under test.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary working directory.
    """
    monkeypatch.chdir(tmp_path)


@asynccontextmanager
async def _open_client(
    qdrant: AsyncQdrantClient,
    reranker: FastEmbedReranker,
    llm: ScriptedLoopLLM,
) -> AsyncGenerator[httpx.AsyncClient]:
    """Start the real ``create_app`` lifespan and yield an ASGI ``httpx.AsyncClient``.

    Args:
        qdrant: In-memory Qdrant client.
        reranker: Cross-encoder reranker.
        llm: Scripted LLM fake.

    Yields:
        Active async test client bound to the real FastAPI app.
    """
    app = create_app(
        APISettings(max_upload_bytes=_UPLOAD_LIMIT),
        IngestionSettings(qdrant_collection=_COLLECTION),
        AgentSettings(max_loop_count=3),
        qdrant_client=qdrant,
        embedder=MockEmbedder(),
        llm=llm,
        reranker=reranker,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        yield http


async def _ingest_sample_paper(http: httpx.AsyncClient) -> str:
    """Upload ``paper-01.pdf`` via ``POST /ingest`` and return its ``document_id``.

    Args:
        http: Async ASGI test client.

    Returns:
        Ingested document ID.
    """
    response = await http.post(
        "/ingest",
        files={"file": (SAMPLE_PAPER_PATH.name, SAMPLE_PAPER_PATH.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["chunks_created"] > 0
    return str(body["document_id"])


async def test_ingest_then_ask_returns_grounded_citations_from_qdrant(
    in_memory_qdrant: AsyncQdrantClient, shared_reranker: FastEmbedReranker
) -> None:
    """Verify ``POST /ingest`` followed by ``POST /ask`` returns citations stored in Qdrant."""
    # Given
    async with _open_client(
        in_memory_qdrant, shared_reranker, ScriptedLoopLLM(grade_verdicts=[True])
    ) as http:
        document_id = await _ingest_sample_paper(http)

        # When
        response = await http.post(
            "/ask", json={"query": "What does this paper study?", "top_k": 3}
        )

    # Then
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What does this paper study?"
    assert body["iterations_taken"] == 1
    assert len(body["citations"]) == 1
    stored_ids = await stored_chunk_ids(in_memory_qdrant, _COLLECTION)
    assert body["citations"][0]["chunk_id"] in stored_ids
    assert body["citations"][0]["document_id"] == document_id


async def test_ask_multi_hop_rewrite_executes_multiple_iterations(
    in_memory_qdrant: AsyncQdrantClient, shared_reranker: FastEmbedReranker
) -> None:
    """Verify ``POST /ask`` runs rewrite + second retrieval when pass 1 is graded insufficient."""
    # Given
    async with _open_client(
        in_memory_qdrant, shared_reranker, ScriptedLoopLLM(grade_verdicts=[False, True])
    ) as http:
        await _ingest_sample_paper(http)

        # When
        response = await http.post("/ask", json={"query": "Explain the architecture", "top_k": 3})

    # Then
    assert response.status_code == 200
    body = response.json()
    assert body["iterations_taken"] == 2
    assert len(body["citations"]) == 1


async def test_ask_maps_llm_error_to_503_through_real_loop(
    in_memory_qdrant: AsyncQdrantClient, shared_reranker: FastEmbedReranker
) -> None:
    """Verify ``LLMError`` raised inside the real loop maps to 503 without leaking details."""
    # Given
    async with _open_client(
        in_memory_qdrant,
        shared_reranker,
        ScriptedLoopLLM(fail_with=LLMError("internal ollama error")),
    ) as http:
        await _ingest_sample_paper(http)

        # When
        response = await http.post("/ask", json={"query": "What does this paper study?"})

    # Then
    assert response.status_code == 503
    assert set(response.json()) == {"detail"}
    assert "internal ollama error" not in response.text


async def test_ask_maps_unknown_citation_to_502_through_real_synthesizer(
    in_memory_qdrant: AsyncQdrantClient, shared_reranker: FastEmbedReranker
) -> None:
    """Verify unknown citation rejection maps to 502 without leaking chunk IDs."""
    # Given
    async with _open_client(
        in_memory_qdrant,
        shared_reranker,
        ScriptedLoopLLM(grade_verdicts=[True], hallucinate_chunk_id="nonexistent-chunk-999"),
    ) as http:
        await _ingest_sample_paper(http)

        # When
        response = await http.post("/ask", json={"query": "What does this paper study?"})

    # Then
    assert response.status_code == 502
    assert set(response.json()) == {"detail"}
    assert "nonexistent-chunk-999" not in response.text
