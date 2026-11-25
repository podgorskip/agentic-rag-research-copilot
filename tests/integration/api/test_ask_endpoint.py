"""``POST /ask`` integration tests exercising the real ``create_app`` lifespan and service wiring.

Qdrant runs in-memory with ``MockEmbedder`` and a scripted ``LLMProtocol`` fake injected at the
leaf adapter boundary, so the real ``IngestionPipeline``, ``RetrievalService``, and
``AgenticLoopService`` wiring runs deterministically without Docker or Ollama.
"""

import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final, TypeVar

import httpx
import pytest
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker
from agentic_rag_copilot.agent import AgentSettings
from agentic_rag_copilot.api import APISettings, create_app
from agentic_rag_copilot.domain.exceptions import LLMError
from agentic_rag_copilot.domain.models import Citation, GradeResult, SynthesizedAnswer
from agentic_rag_copilot.domain.protocols import LLMProtocol
from agentic_rag_copilot.ingestion import IngestionSettings
from tests.mocks.ingestion import MockEmbedder

_PAPER_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "e2e" / "data" / "paper-01.pdf"
_COLLECTION: Final[str] = "integration_ask_endpoint"
_UPLOAD_LIMIT: Final[int] = 10 * 1024 * 1024
_CHUNK_ID_RE: Final[re.Pattern[str]] = re.compile(r"chunk_id=([^\s\]]+)")

_T = TypeVar("_T", bound=BaseModel)


class _ScriptedAskLLM(LLMProtocol):
    """Minimal scripted LLM returning preset grade verdicts and a grounded citation."""

    def __init__(
        self,
        grade_verdicts: list[bool] | None = None,
        *,
        fail_with: Exception | None = None,
        hallucinate_chunk_id: str | None = None,
    ) -> None:
        self._verdicts = iter(grade_verdicts or [True])
        self._fail_with = fail_with
        self._hallucinate_chunk_id = hallucinate_chunk_id

    async def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        if self._fail_with is not None:
            raise self._fail_with
        return "Retrieval-Augmented Generation parametric non-parametric memory"

    async def generate_structured(
        self, prompt: str, response_model: type[_T], system_prompt: str | None = None
    ) -> _T:
        if self._fail_with is not None:
            raise self._fail_with
        if response_model is GradeResult:
            verdict = next(self._verdicts, True)
            payload: BaseModel = GradeResult(is_relevant=verdict, reasoning="scripted")
        else:
            match = _CHUNK_ID_RE.search(prompt)
            chunk_id = self._hallucinate_chunk_id or (match.group(1) if match else "missing-id")
            payload = SynthesizedAnswer(
                answer="The ingested paper describes the requested topic [1].",
                citations=[
                    Citation(
                        chunk_id=chunk_id,
                        document_id="placeholder-doc",
                        source_uri="placeholder-uri",
                        quote="supporting excerpt from paper",
                    )
                ],
            )
        return response_model.model_validate(payload.model_dump())


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own ``config/.env`` from influencing the app under test.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary working directory.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="session")
def reranker() -> FastEmbedReranker:
    """Reuse a single ``FastEmbedReranker`` across integration tests.

    Returns:
        Configured ``FastEmbedReranker``.
    """
    return FastEmbedReranker()


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    """Provide an isolated in-memory Qdrant client per test.

    Yields:
        ``AsyncQdrantClient`` backed by in-memory storage.
    """
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


@asynccontextmanager
async def _open_client(
    qdrant: AsyncQdrantClient,
    reranker: FastEmbedReranker,
    llm: _ScriptedAskLLM,
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
        files={"file": (_PAPER_PATH.name, _PAPER_PATH.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["chunks_created"] > 0
    return str(body["document_id"])


async def _stored_chunk_ids(qdrant: AsyncQdrantClient) -> set[str]:
    """Collect all ``chunk_id`` payload values currently stored in the test collection.

    Args:
        qdrant: In-memory Qdrant client.

    Returns:
        Set of stored chunk IDs.
    """
    points, _ = await qdrant.scroll(_COLLECTION, limit=500, with_payload=True)
    return {
        str(point.payload["chunk_id"])
        for point in points
        if point.payload is not None and "chunk_id" in point.payload
    }


async def test_ingest_then_ask_returns_grounded_citations_from_qdrant(
    qdrant: AsyncQdrantClient, reranker: FastEmbedReranker
) -> None:
    """Verify ``POST /ingest`` followed by ``POST /ask`` returns citations stored in Qdrant."""
    # Given
    async with _open_client(qdrant, reranker, _ScriptedAskLLM(grade_verdicts=[True])) as http:
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
    stored_ids = await _stored_chunk_ids(qdrant)
    assert body["citations"][0]["chunk_id"] in stored_ids
    assert body["citations"][0]["document_id"] == document_id


async def test_ask_multi_hop_rewrite_executes_multiple_iterations(
    qdrant: AsyncQdrantClient, reranker: FastEmbedReranker
) -> None:
    """Verify ``POST /ask`` runs rewrite + second retrieval when pass 1 is graded insufficient."""
    # Given
    async with _open_client(
        qdrant, reranker, _ScriptedAskLLM(grade_verdicts=[False, True])
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
    qdrant: AsyncQdrantClient, reranker: FastEmbedReranker
) -> None:
    """Verify ``LLMError`` raised inside the real loop maps to 503 without leaking details."""
    # Given
    async with _open_client(
        qdrant, reranker, _ScriptedAskLLM(fail_with=LLMError("internal ollama error"))
    ) as http:
        await _ingest_sample_paper(http)

        # When
        response = await http.post("/ask", json={"query": "What does this paper study?"})

    # Then
    assert response.status_code == 503
    assert set(response.json()) == {"detail"}
    assert "internal ollama error" not in response.text


async def test_ask_maps_unknown_citation_to_502_through_real_synthesizer(
    qdrant: AsyncQdrantClient, reranker: FastEmbedReranker
) -> None:
    """Verify unknown citation rejection maps to 502 without leaking chunk IDs."""
    # Given
    async with _open_client(
        qdrant,
        reranker,
        _ScriptedAskLLM(grade_verdicts=[True], hallucinate_chunk_id="nonexistent-chunk-999"),
    ) as http:
        await _ingest_sample_paper(http)

        # When
        response = await http.post("/ask", json={"query": "What does this paper study?"})

    # Then
    assert response.status_code == 502
    assert set(response.json()) == {"detail"}
    assert "nonexistent-chunk-999" not in response.text
