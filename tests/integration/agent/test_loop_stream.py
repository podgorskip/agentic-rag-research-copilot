"""Integration tests for ``AgenticLoopService.ask_stream`` using the real ``create_app`` wiring.

Qdrant runs in-memory with ``MockEmbedder`` and ``ScriptedLoopLLM`` injected at the leaf adapter
boundary, so the real ``IngestionPipeline``, ``RetrievalService``, and ``AgenticLoopService``
wiring runs deterministically without Docker or Ollama.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

import pytest
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker
from agentic_rag_copilot.agent import AgenticLoopService, AgentSettings
from agentic_rag_copilot.api import APISettings, create_app
from agentic_rag_copilot.domain.exceptions import LLMError
from agentic_rag_copilot.domain.models import AskRequest, IngestRequest, SourceType
from agentic_rag_copilot.ingestion import IngestionSettings
from tests.integration.conftest import SAMPLE_PAPER_PATH, stored_chunk_ids
from tests.mocks.agent import ScriptedLoopLLM
from tests.mocks.ingestion import MockEmbedder

_COLLECTION: Final[str] = "integration_loop_stream"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own ``config/.env`` from influencing the app under test.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary working directory.
    """
    monkeypatch.chdir(tmp_path)


@asynccontextmanager
async def _open_seeded_loop(
    qdrant: AsyncQdrantClient,
    reranker: FastEmbedReranker,
    llm: ScriptedLoopLLM,
) -> AsyncGenerator[tuple[AgenticLoopService, set[str]]]:
    """Start the real ``create_app`` lifespan, ingest ``paper-01.pdf``, and yield the loop service.

    Args:
        qdrant: In-memory Qdrant client.
        reranker: Cross-encoder reranker.
        llm: Scripted LLM fake.

    Yields:
        Tuple of ``(AgenticLoopService, stored_chunk_ids)``.
    """
    app = create_app(
        APISettings(),
        IngestionSettings(qdrant_collection=_COLLECTION),
        AgentSettings(max_loop_count=3),
        qdrant_client=qdrant,
        embedder=MockEmbedder(),
        llm=llm,
        reranker=reranker,
    )
    async with app.router.lifespan_context(app):
        ingest_response = await app.state.ingestion_pipeline.ingest(
            IngestRequest(source_uri=str(SAMPLE_PAPER_PATH), source_type=SourceType.PDF)
        )
        assert ingest_response.chunks_created > 0
        stored_ids = await stored_chunk_ids(qdrant, _COLLECTION)
        yield app.state.agentic_loop_service, stored_ids


async def test_ask_stream_single_hop_grounds_citations_in_qdrant(
    in_memory_qdrant: AsyncQdrantClient, shared_reranker: FastEmbedReranker
) -> None:
    """Verify single-hop ``ask_stream`` yields ordered events and real Qdrant citations."""
    # Given
    request = AskRequest(query="What does this paper study?", top_k=3)
    async with _open_seeded_loop(
        in_memory_qdrant, shared_reranker, ScriptedLoopLLM(grade_verdicts=[True])
    ) as (loop_service, stored_ids):
        # When
        events = [event async for event in loop_service.ask_stream(request)]

    # Then
    assert [event.type for event in events] == ["retrieve", "grade", "generate", "answer"]
    answer_payload = events[-1].payload
    assert answer_payload["iterations_taken"] == 1
    assert len(answer_payload["citations"]) == 1
    assert answer_payload["citations"][0]["chunk_id"] in stored_ids


async def test_ask_stream_two_rewrites_accumulates_and_grounds_answer(
    in_memory_qdrant: AsyncQdrantClient, shared_reranker: FastEmbedReranker
) -> None:
    """Verify 2-rewrite multi-hop ``ask_stream`` executes 3 passes and emits full sequence."""
    # Given
    request = AskRequest(query="Explain the architecture", top_k=3)
    async with _open_seeded_loop(
        in_memory_qdrant, shared_reranker, ScriptedLoopLLM(grade_verdicts=[False, False, True])
    ) as (loop_service, stored_ids):
        # When
        events = [event async for event in loop_service.ask_stream(request)]

    # Then
    assert [event.type for event in events] == [
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
        "answer",
    ]
    assert events[-1].payload["iterations_taken"] == 3
    assert events[-1].payload["citations"][0]["chunk_id"] in stored_ids


async def test_ask_stream_emits_error_event_on_real_loop_llm_failure(
    in_memory_qdrant: AsyncQdrantClient, shared_reranker: FastEmbedReranker
) -> None:
    """Verify ``LLMError`` inside the real loop emits an ``error`` event and closes cleanly."""
    # Given
    request = AskRequest(query="What does this paper study?", top_k=3)
    async with _open_seeded_loop(
        in_memory_qdrant, shared_reranker, ScriptedLoopLLM(fail_with=LLMError("ollama down"))
    ) as (loop_service, _):
        # When
        events = [event async for event in loop_service.ask_stream(request)]

    # Then
    assert [event.type for event in events] == ["retrieve", "error"]
    assert events[-1].payload["error_type"] == "LLMError"
    assert "ollama down" not in str(events[-1].payload)
