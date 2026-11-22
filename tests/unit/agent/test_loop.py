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


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc-1", text=f"text for {chunk_id}")


class ScriptedRetrievalService(RetrievalService):
    """Returns one scripted chunk list per call and records each query and ``final_top_k``."""

    def __init__(self, results: list[list[Chunk]]) -> None:
        self._results = list(results)
        self.calls: list[tuple[Query, int]] = []

    async def retrieve(
        self, query: Query, initial_top_k: int = 20, final_top_k: int = 5
    ) -> list[Chunk]:
        self.calls.append((query, final_top_k))
        return self._results[len(self.calls) - 1]


class ScriptedGrader(RelevanceGrader):
    """Returns scripted grades in order and snapshots the chunk IDs seen on each call."""

    def __init__(self, verdicts: list[bool]) -> None:
        super().__init__(MockLLM())
        self._verdicts = list(verdicts)
        self.seen_chunk_ids: list[list[str]] = []

    async def grade(self, state: AgentState) -> GradeResult:
        self.seen_chunk_ids.append([chunk.id for chunk in state.retrieved_chunks])
        verdict = self._verdicts[len(self.seen_chunk_ids) - 1]
        return GradeResult(is_relevant=verdict, reasoning=f"pass {len(self.seen_chunk_ids)}")


class ScriptedRewriter(QueryRewriter):
    """Returns scripted rewrites in order and records the reasoning and history it saw."""

    def __init__(self, rewrites: list[str]) -> None:
        super().__init__(MockLLM())
        self._rewrites = list(rewrites)
        self.calls: list[tuple[str | None, list[str]]] = []

    async def rewrite(self, state: AgentState, reasoning: str | None = None) -> str:
        self.calls.append((reasoning, list(state.query_history)))
        return self._rewrites[len(self.calls) - 1]


class RecordingSynthesizer(AnswerSynthesizer):
    """Cites every accumulated chunk and records the state snapshot it was called with."""

    def __init__(self) -> None:
        super().__init__(MockLLM())
        self.states: list[AgentState] = []

    async def synthesize(self, state: AgentState) -> SynthesizedAnswer:
        self.states.append(state.model_copy(deep=True))
        return SynthesizedAnswer(
            answer=f"answer from {len(state.retrieved_chunks)} chunks",
            citations=[
                Citation(chunk_id=c.id, document_id=c.document_id, source_uri="p", quote=c.text)
                for c in state.retrieved_chunks
            ],
        )


def _build(
    retrieval: ScriptedRetrievalService,
    grader: ScriptedGrader,
    rewriter: ScriptedRewriter,
    synthesizer: RecordingSynthesizer,
    max_loop_count: int = 3,
) -> tuple[AgenticLoopService, MagicMock]:
    langfuse = create_autospec(Langfuse, instance=True)
    service = AgenticLoopService(
        retrieval_service=retrieval,
        grader=grader,
        rewriter=rewriter,
        synthesizer=synthesizer,
        settings=AgentSettings(max_loop_count=max_loop_count),
        langfuse_client=cast("Langfuse", langfuse),
    )
    return service, langfuse


def _span_names(langfuse: MagicMock) -> list[str]:
    return [call.kwargs["name"] for call in langfuse.start_as_current_span.call_args_list]


async def test_ask_single_hop_sufficient_on_first_try() -> None:
    # Given
    retrieval = ScriptedRetrievalService([[_chunk("c1"), _chunk("c2")]])
    grader = ScriptedGrader([True])
    rewriter = ScriptedRewriter([])
    synthesizer = RecordingSynthesizer()
    service, langfuse = _build(retrieval, grader, rewriter, synthesizer)

    # When
    response = await service.ask(
        AskRequest(query="What is RAG?", top_k=3, filters={"document_id": "doc-1"})
    )

    # Then
    assert response.query == "What is RAG?"
    assert response.answer == "answer from 2 chunks"
    assert [c.chunk_id for c in response.citations] == ["c1", "c2"]
    assert response.iterations_taken == 1
    query, final_top_k = retrieval.calls[0]
    assert query.text == "What is RAG?"
    assert query.filters == {"document_id": "doc-1"}
    assert final_top_k == 3
    assert rewriter.calls == []
    final_state = synthesizer.states[0]
    assert final_state.is_context_sufficient is True
    assert final_state.loop_count == 0
    assert final_state.query_history == ["What is RAG?"]
    assert _span_names(langfuse) == ["agentic_loop", "grade", "generate"]


async def test_ask_multi_hop_accumulates_deduplicated_chunks_across_two_rewrites() -> None:
    # Given
    retrieval = ScriptedRetrievalService(
        [
            [_chunk("c1"), _chunk("c2")],
            [_chunk("c2"), _chunk("c3")],
            [_chunk("c3"), _chunk("c1"), _chunk("c4")],
        ]
    )
    grader = ScriptedGrader([False, False, True])
    rewriter = ScriptedRewriter(["rewrite one", "rewrite two"])
    synthesizer = RecordingSynthesizer()
    service, langfuse = _build(retrieval, grader, rewriter, synthesizer, max_loop_count=3)

    # When
    response = await service.ask(AskRequest(query="original question"))

    # Then
    assert response.iterations_taken == 3
    assert response.query == "original question"
    assert [q.text for q, _ in retrieval.calls] == [
        "original question",
        "rewrite one",
        "rewrite two",
    ]
    assert grader.seen_chunk_ids == [
        ["c1", "c2"],
        ["c1", "c2", "c3"],
        ["c1", "c2", "c3", "c4"],
    ]
    assert rewriter.calls == [
        ("pass 1", ["original question"]),
        ("pass 2", ["original question", "rewrite one"]),
    ]
    final_state = synthesizer.states[0]
    assert final_state.current_query == "rewrite two"
    assert final_state.query_history == ["original question", "rewrite one", "rewrite two"]
    assert final_state.loop_count == 2
    assert final_state.is_context_sufficient is True
    assert [c.chunk_id for c in response.citations] == ["c1", "c2", "c3", "c4"]
    assert _span_names(langfuse) == [
        "agentic_loop",
        "grade",
        "rewrite",
        "grade",
        "rewrite",
        "grade",
        "generate",
    ]


async def test_ask_returns_best_effort_answer_when_max_loop_count_exhausted() -> None:
    # Given
    retrieval = ScriptedRetrievalService([[_chunk("c1")], [], [_chunk("c2")]])
    grader = ScriptedGrader([False, False, False])
    rewriter = ScriptedRewriter(["rewrite one", "rewrite two"])
    synthesizer = RecordingSynthesizer()
    service, _ = _build(retrieval, grader, rewriter, synthesizer, max_loop_count=2)

    # When
    response = await service.ask(AskRequest(query="unanswerable"))

    # Then
    assert response.iterations_taken == 3
    assert len(retrieval.calls) == 3
    assert len(rewriter.calls) == 2
    assert len(synthesizer.states) == 1
    final_state = synthesizer.states[0]
    assert final_state.is_context_sufficient is False
    assert final_state.loop_count == 2
    assert [c.id for c in final_state.retrieved_chunks] == ["c1", "c2"]
    assert response.answer == "answer from 2 chunks"


async def test_ask_records_latency_metadata_on_step_spans() -> None:
    # Given
    retrieval = ScriptedRetrievalService([[_chunk("c1")], [_chunk("c2")]])
    service, langfuse = _build(
        retrieval,
        ScriptedGrader([False, True]),
        ScriptedRewriter(["better query"]),
        RecordingSynthesizer(),
    )

    # When
    await service.ask(AskRequest(query="q"))

    # Then
    span = langfuse.start_as_current_span.return_value.__enter__.return_value
    metadata = [call.kwargs["metadata"] for call in span.update.call_args_list]
    step_metadata = metadata[:-1]
    assert len(step_metadata) == 4  # grade, rewrite, grade, generate
    assert all(isinstance(m["latency_ms"], float) for m in step_metadata)
    loop_metadata = metadata[-1]
    assert loop_metadata["iterations_taken"] == 2
    assert loop_metadata["rewrites"] == 1
    assert loop_metadata["query_history"] == ["q", "better query"]


async def test_ask_propagates_llm_error_from_grader() -> None:
    # Given
    class FailingGrader(ScriptedGrader):
        async def grade(self, state: AgentState) -> GradeResult:
            raise LLMError("Ollama unreachable")

    service, _ = _build(
        ScriptedRetrievalService([[_chunk("c1")]]),
        FailingGrader([]),
        ScriptedRewriter([]),
        RecordingSynthesizer(),
    )

    # When / Then
    with pytest.raises(LLMError, match="Ollama unreachable"):
        await service.ask(AskRequest(query="q"))


async def test_ask_flags_repeated_rewrite_on_rewrite_span() -> None:
    # Given
    retrieval = ScriptedRetrievalService([[_chunk("c1")], [_chunk("c1")]])
    service, langfuse = _build(
        retrieval,
        ScriptedGrader([False, False]),
        ScriptedRewriter(["same question"]),
        RecordingSynthesizer(),
        max_loop_count=1,
    )

    # When
    response = await service.ask(AskRequest(query="same question"))

    # Then
    assert response.iterations_taken == 2
    span = langfuse.start_as_current_span.return_value.__enter__.return_value
    rewrite_metadata = span.update.call_args_list[1].kwargs["metadata"]
    assert rewrite_metadata["is_repeat"] is True
