"""Integration tests for QueryRewriter against local Ollama and arXiv:2005.11401."""

import pytest

from agentic_rag_copilot.adapters.llm import OllamaLLM
from agentic_rag_copilot.agent import QueryRewriter, RelevanceGrader
from agentic_rag_copilot.domain.models import AgentState, Query
from agentic_rag_copilot.services import RetrievalService
from tests.integration.agent.conftest import AGENT_SETTINGS, ollama_is_reachable

pytestmark = pytest.mark.skipif(
    not ollama_is_reachable(),
    reason=f"Ollama is not reachable at {AGENT_SETTINGS.ollama_url}; start it to run this test",
)


async def test_rewriter_recovers_query_after_irrelevant_initial_retrieval(
    ollama_llm: OllamaLLM,
    retrieval_service: RetrievalService,
) -> None:
    """Verify rewriting an off-target query recovers relevant chunks that pass grading."""
    # Given
    original_query = (
        "What is the difference between parametric memory and non-parametric memory in RAG?"
    )
    initial_search_query = "Broader Impact societal benefits and ethical risks"
    initial_chunks = await retrieval_service.retrieve(
        Query(text=initial_search_query),
        initial_top_k=30,
        final_top_k=6,
    )
    assert initial_chunks
    grader = RelevanceGrader(ollama_llm)
    rewriter = QueryRewriter(ollama_llm)
    state = AgentState(
        original_query=original_query,
        current_query=original_query,
        query_history=[initial_search_query],
        retrieved_chunks=initial_chunks,
    )

    # When
    initial_grade = await grader.grade(state)

    # Then
    assert initial_grade.is_relevant is False
    assert initial_grade.reasoning.strip() != ""

    # Given
    state.current_query = initial_search_query

    # When
    rewritten_query = await rewriter.rewrite(state, reasoning=initial_grade.reasoning)

    # Then
    assert rewritten_query.strip() != ""
    assert rewritten_query.lower() != initial_search_query.lower()

    # Given
    state.query_history.append(rewritten_query)
    state.current_query = rewritten_query
    recovered_chunks = await retrieval_service.retrieve(
        Query(text=rewritten_query),
        initial_top_k=30,
        final_top_k=6,
    )
    assert recovered_chunks
    state.retrieved_chunks = recovered_chunks

    # When
    recovered_grade = await grader.grade(state)

    # Then
    assert recovered_grade.is_relevant is True, (
        f"Expected relevant grade after rewrite {rewritten_query!r}; "
        f"reasoning: {recovered_grade.reasoning!r}"
    )
    assert recovered_grade.reasoning.strip() != ""
