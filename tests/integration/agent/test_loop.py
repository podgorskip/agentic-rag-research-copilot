"""End-to-end integration tests for AgenticLoopService against local Ollama and arXiv:2005.11401.

Wires the real ``OllamaLLM``, real ``RetrievalService`` (in-memory Qdrant + FastEmbed reranker),
``RelevanceGrader``, ``QueryRewriter`` and ``AnswerSynthesizer``. Skipped when Ollama is not
reachable.
"""

import pytest
from langfuse import Langfuse

from agentic_rag_copilot.adapters.llm import OllamaLLM
from agentic_rag_copilot.agent import (
    AgenticLoopService,
    AgentSettings,
    AnswerSynthesizer,
    QueryRewriter,
    RelevanceGrader,
)
from agentic_rag_copilot.domain.models import (
    AgentState,
    AskRequest,
    AskResponse,
    Chunk,
    GradeResult,
)
from agentic_rag_copilot.services import RetrievalService
from tests.integration.agent.conftest import AGENT_SETTINGS, ollama_is_reachable

pytestmark = pytest.mark.skipif(
    not ollama_is_reachable(),
    reason=f"Ollama is not reachable at {AGENT_SETTINGS.ollama_url}; start it to run this test",
)


class _RecordingGrader(RelevanceGrader):
    """Real grader that additionally records every verdict, to check grading gated generation."""

    def __init__(self, llm: OllamaLLM) -> None:
        super().__init__(llm)
        self.results: list[GradeResult] = []

    async def grade(self, state: AgentState) -> GradeResult:
        result = await super().grade(state)
        self.results.append(result)
        return result


def _build_loop(
    llm: OllamaLLM, retrieval_service: RetrievalService, max_loop_count: int
) -> tuple[AgenticLoopService, _RecordingGrader]:
    grader = _RecordingGrader(llm)
    service = AgenticLoopService(
        retrieval_service=retrieval_service,
        grader=grader,
        rewriter=QueryRewriter(llm),
        synthesizer=AnswerSynthesizer(llm),
        settings=AgentSettings(max_loop_count=max_loop_count),
        langfuse_client=Langfuse(tracing_enabled=False),
    )
    return service, grader


def _assert_grounded(response: AskResponse, corpus: list[Chunk]) -> None:
    chunks_by_id = {chunk.id: chunk for chunk in corpus}
    for citation in response.citations:
        assert citation.chunk_id in chunks_by_id, f"Ungrounded citation {citation.chunk_id!r}"
        assert citation.document_id == chunks_by_id[citation.chunk_id].document_id


def _assert_grading_gated_generation(
    response: AskResponse, grader: _RecordingGrader, max_loop_count: int
) -> None:
    verdicts = [result.is_relevant for result in grader.results]
    assert len(verdicts) == response.iterations_taken
    assert not any(verdicts[:-1]), f"Loop continued after a sufficient grade: {verdicts}"
    assert verdicts[-1] or response.iterations_taken == max_loop_count + 1


async def test_loop_answers_conceptual_question_with_grounded_citations(
    ollama_llm: OllamaLLM,
    retrieval_service: RetrievalService,
    embedded_validation_chunks: list[Chunk],
) -> None:
    """Verify an on-topic question yields a non-empty answer citing real corpus chunks."""
    # Given
    max_loop_count = 3
    service, grader = _build_loop(ollama_llm, retrieval_service, max_loop_count)
    request = AskRequest(
        query="What is the difference between parametric memory and non-parametric memory in RAG?",
        top_k=6,
    )

    # When
    response = await service.ask(request)

    # Then
    assert response.query == request.query
    assert response.answer.strip() != ""
    assert response.citations, "Expected at least one citation for an on-topic question"
    _assert_grounded(response, embedded_validation_chunks)
    _assert_grading_gated_generation(response, grader, max_loop_count)


async def test_loop_multi_hop_question_requires_rewrites_and_stays_grounded(
    ollama_llm: OllamaLLM,
    retrieval_service: RetrievalService,
    embedded_validation_chunks: list[Chunk],
) -> None:
    """Verify a Milestone 03 ``multi_hop`` question takes more than one retrieve/grade pass.

    ``top_k=2`` keeps each hop narrow so the two halves of the question (frozen vs. fine-tuned
    document encoder) are unlikely to both be covered by the first retrieval, forcing the loop
    to rewrite and accumulate context across hops.
    """
    # Given
    max_loop_count = 3
    service, grader = _build_loop(ollama_llm, retrieval_service, max_loop_count)
    request = AskRequest(
        query="What happens during RAG training when the document encoder is frozen vs fine-tuned?",
        top_k=2,
    )

    # When
    response = await service.ask(request)

    # Then
    assert response.iterations_taken > 1, (
        "Expected at least one rewrite; grader verdicts: "
        f"{[(r.is_relevant, r.reasoning) for r in grader.results]}"
    )
    assert response.answer.strip() != ""
    _assert_grounded(response, embedded_validation_chunks)
    _assert_grading_gated_generation(response, grader, max_loop_count)


async def test_loop_returns_best_effort_answer_for_off_topic_question(
    ollama_llm: OllamaLLM,
    retrieval_service: RetrievalService,
    embedded_validation_chunks: list[Chunk],
) -> None:
    """Verify an unanswerable question exhausts ``max_loop_count`` and still returns."""
    # Given
    max_loop_count = 1
    service, grader = _build_loop(ollama_llm, retrieval_service, max_loop_count)

    # When
    response = await service.ask(
        AskRequest(query="How does Kubernetes horizontal pod autoscaling calculate replicas?")
    )

    # Then
    assert response.iterations_taken == max_loop_count + 1
    assert all(result.is_relevant is False for result in grader.results)
    assert response.answer.strip() != ""
    _assert_grounded(response, embedded_validation_chunks)
