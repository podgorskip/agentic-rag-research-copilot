"""Unit tests for ``AgenticLoopService.ask_stream`` and ``AgentEvent``."""

from typing import cast
from unittest.mock import MagicMock, create_autospec

import pytest
from langfuse import Langfuse

from agentic_rag_copilot.agent import (
    AgenticLoopService,
    AgentSettings,
    AnswerSynthesizer,
    QueryRewriter,
    RelevanceGrader,
)
from agentic_rag_copilot.domain.exceptions import LLMError
from agentic_rag_copilot.domain.models import (
    AgentEvent,
    AgentState,
    AskRequest,
    Chunk,
    Citation,
    GradeResult,
    Query,
    SynthesizedAnswer,
)
from agentic_rag_copilot.services import RetrievalService
from tests.mocks.agent import MockLLM
from tests.mocks.ingestion import InMemoryVectorStore, MockEmbedder
from tests.mocks.retrieval import RecordingReranker


def _chunk(chunk_id: str) -> Chunk:
    """Create a minimal ``Chunk`` for streaming loop unit tests.

    Args:
        chunk_id: Unique chunk identifier.

    Returns:
        ``Chunk`` bound to ``doc-1``.
    """
    return Chunk(id=chunk_id, document_id="doc-1", text=f"text for {chunk_id}")


class _ScriptedRetrieval(RetrievalService):
    """Returns one scripted chunk list per retrieval pass."""

    def __init__(self, results: list[list[Chunk]]) -> None:
        super().__init__(
            embedder=MockEmbedder(),
            vector_store=InMemoryVectorStore(),
            reranker=RecordingReranker([]),
            langfuse_client=_fake_langfuse(),
        )
        self._results = list(results)
        self._calls = 0

    async def retrieve(
        self, query: Query, initial_top_k: int = 20, final_top_k: int = 5
    ) -> list[Chunk]:
        chunks = self._results[self._calls]
        self._calls += 1
        return chunks


class _ScriptedGrader(RelevanceGrader):
    """Returns scripted grade verdicts or raises a configured exception."""

    def __init__(self, verdicts: list[bool], *, fail_with: Exception | None = None) -> None:
        super().__init__(MockLLM())
        self._verdicts = list(verdicts)
        self._fail_with = fail_with
        self._calls = 0

    async def grade(self, state: AgentState) -> GradeResult:
        if self._fail_with is not None:
            raise self._fail_with
        verdict = self._verdicts[self._calls]
        self._calls += 1
        return GradeResult(is_relevant=verdict, reasoning=f"pass {self._calls}")


class _ScriptedRewriter(QueryRewriter):
    """Returns scripted query rewrites in sequence."""

    def __init__(self, rewrites: list[str]) -> None:
        super().__init__(MockLLM())
        self._rewrites = list(rewrites)
        self._calls = 0

    async def rewrite(self, state: AgentState, reasoning: str | None = None) -> str:
        rewritten = self._rewrites[self._calls]
        self._calls += 1
        return rewritten


class _ScriptedSynthesizer(AnswerSynthesizer):
    """Returns a grounded answer citing every accumulated chunk or raises ``fail_with``."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        super().__init__(MockLLM())
        self._fail_with = fail_with

    async def synthesize(self, state: AgentState) -> SynthesizedAnswer:
        if self._fail_with is not None:
            raise self._fail_with
        citations = [
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                source_uri="https://example.org/doc",
                quote=chunk.text,
            )
            for chunk in state.retrieved_chunks
        ]
        return SynthesizedAnswer(
            answer=f"answer from {len(state.retrieved_chunks)} chunks",
            citations=citations,
        )


def _fake_langfuse() -> Langfuse:
    """Create a no-op Langfuse mock supporting context-manager spans.

    Returns:
        Mocked ``Langfuse`` client.
    """
    client = create_autospec(Langfuse, instance=True)
    span = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = span
    cm.__exit__.return_value = None
    client.start_as_current_span.return_value = cm
    return cast("Langfuse", client)


def _build_service(
    results: list[list[Chunk]],
    verdicts: list[bool],
    rewrites: list[str] | None = None,
    *,
    max_loop_count: int = 3,
    grader_error: Exception | None = None,
    synthesizer_error: Exception | None = None,
) -> AgenticLoopService:
    """Assemble an ``AgenticLoopService`` wired with scripted unit-test doubles.

    Args:
        results: Chunk lists returned per retrieval pass.
        verdicts: Grader relevance verdicts per pass.
        rewrites: Rewritten queries returned per rewrite call.
        max_loop_count: Maximum rewrite iterations.
        grader_error: Optional exception raised by the grader.
        synthesizer_error: Optional exception raised by the synthesizer.

    Returns:
        Configured ``AgenticLoopService``.
    """
    return AgenticLoopService(
        retrieval_service=_ScriptedRetrieval(results),
        grader=_ScriptedGrader(verdicts, fail_with=grader_error),
        rewriter=_ScriptedRewriter(rewrites or []),
        synthesizer=_ScriptedSynthesizer(fail_with=synthesizer_error),
        settings=AgentSettings(max_loop_count=max_loop_count),
        langfuse_client=_fake_langfuse(),
    )


async def test_ask_stream_single_hop_emits_expected_event_order_and_payloads() -> None:
    """Verify single-hop ``ask_stream`` yields retrieve -> grade -> generate -> answer."""
    # Given
    service = _build_service([[_chunk("c1"), _chunk("c2")]], verdicts=[True])
    request = AskRequest(query="What is RAG?", top_k=5)

    # When
    events = [event async for event in service.ask_stream(request)]

    # Then
    assert [event.type for event in events] == ["retrieve", "grade", "generate", "answer"]
    assert [event.iteration for event in events] == [1, 1, 1, 1]
    assert events[0].payload["query"] == "What is RAG?"
    assert events[1].payload == {"is_context_sufficient": True, "reasoning": "pass 1"}
    assert events[2].payload == {"query": "What is RAG?", "chunk_count": 2}
    assert events[3].payload["query"] == "What is RAG?"
    assert events[3].payload["iterations_taken"] == 1
    assert [c["chunk_id"] for c in events[3].payload["citations"]] == ["c1", "c2"]


async def test_ask_stream_two_rewrite_multi_hop_emits_expected_event_order() -> None:
    """Verify a 2-rewrite loop yields the exact 10-event sequence with incrementing iterations."""
    # Given
    service = _build_service(
        [[_chunk("c1")], [_chunk("c2")], [_chunk("c3")]],
        verdicts=[False, False, True],
        rewrites=["rewrite one", "rewrite two"],
    )
    request = AskRequest(query="original question")

    # When
    events = [event async for event in service.ask_stream(request)]

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
    assert [event.iteration for event in events] == [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    assert events[2].payload == {"query": "rewrite one"}
    assert events[3].payload["query"] == "rewrite one"
    assert events[5].payload == {"query": "rewrite two"}
    assert events[6].payload["query"] == "rewrite two"
    assert events[-1].payload["iterations_taken"] == 3


async def test_ask_stream_answer_payload_equals_ask_response() -> None:
    """Verify ``ask_stream`` terminal answer payload matches ``ask()`` for identical inputs."""
    # Given
    request = AskRequest(query="compare RAG-Token and RAG-Sequence", top_k=4)
    stream_service = _build_service(
        [[_chunk("c1")], [_chunk("c2")]],
        verdicts=[False, True],
        rewrites=["RAG-Token vs RAG-Sequence"],
    )
    ask_service = _build_service(
        [[_chunk("c1")], [_chunk("c2")]],
        verdicts=[False, True],
        rewrites=["RAG-Token vs RAG-Sequence"],
    )

    # When
    events = [event async for event in stream_service.ask_stream(request)]
    response = await ask_service.ask(request)

    # Then
    answer_event = events[-1]
    assert answer_event.type == "answer"
    assert answer_event.payload == response.model_dump(mode="json")


async def test_ask_stream_yields_error_event_while_ask_reraises_llm_error() -> None:
    """Verify ``ask_stream`` yields ``error`` on ``LLMError`` while ``ask`` re-raises."""
    # Given
    request = AskRequest(query="What is DPR?")
    stream_service = _build_service(
        [[_chunk("c1")]],
        verdicts=[],
        grader_error=LLMError("Ollama unreachable"),
    )
    ask_service = _build_service(
        [[_chunk("c1")]],
        verdicts=[],
        grader_error=LLMError("Ollama unreachable"),
    )

    # When
    events = [event async for event in stream_service.ask_stream(request)]

    # Then
    assert [event.type for event in events] == ["retrieve", "error"]
    assert events[-1].iteration == 1
    assert events[-1].payload["error_type"] == "LLMError"
    assert "Ollama unreachable" not in str(events[-1].payload)

    # When / Then
    with pytest.raises(LLMError, match="Ollama unreachable"):
        await ask_service.ask(request)


async def test_ask_stream_yields_error_event_while_ask_reraises_value_error() -> None:
    """Verify ``ask_stream`` yields ``error`` on synthesizer ``ValueError`` while ``ask`` raises."""
    # Given
    request = AskRequest(query="What is DPR?")
    stream_service = _build_service(
        [[_chunk("c1")]],
        verdicts=[True],
        synthesizer_error=ValueError("Unknown citation chunk_id"),
    )
    ask_service = _build_service(
        [[_chunk("c1")]],
        verdicts=[True],
        synthesizer_error=ValueError("Unknown citation chunk_id"),
    )

    # When
    events = [event async for event in stream_service.ask_stream(request)]

    # Then
    assert [event.type for event in events] == ["retrieve", "grade", "generate", "error"]
    assert events[-1].payload["error_type"] == "ValueError"
    assert "Unknown citation" not in str(events[-1].payload)

    # When / Then
    with pytest.raises(ValueError, match="Unknown citation"):
        await ask_service.ask(request)


def test_agent_event_json_schema_documents_fields() -> None:
    """Verify ``AgentEvent`` exposes code-first JSON schema metadata for OpenAPI consumers."""
    # When
    schema = AgentEvent.model_json_schema()

    # Then
    assert set(schema["properties"]) == {"type", "iteration", "payload"}
    assert schema["properties"]["type"]["description"]
    assert schema["properties"]["iteration"]["minimum"] == 0
    assert schema["properties"]["payload"]["description"]
