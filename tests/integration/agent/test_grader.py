"""Integration tests for RelevanceGrader against local Ollama and arXiv:2005.11401."""

import pytest

from agentic_rag_copilot.adapters.llm import OllamaLLM
from agentic_rag_copilot.agent import RelevanceGrader
from agentic_rag_copilot.domain.models import AgentState, Query
from agentic_rag_copilot.services import RetrievalService
from tests.integration.agent.conftest import AGENT_SETTINGS, ollama_is_reachable

pytestmark = pytest.mark.skipif(
    not ollama_is_reachable(),
    reason=f"Ollama is not reachable at {AGENT_SETTINGS.ollama_url}; start it to run this test",
)


async def test_grader_flags_empty_chunk_set_as_insufficient(ollama_llm: OllamaLLM) -> None:
    """Verify grader marks an empty chunk list as insufficient."""
    # Given
    grader = RelevanceGrader(ollama_llm)
    state = AgentState(
        original_query="What is the difference between parametric and non-parametric memory?",
        current_query="What is the difference between parametric and non-parametric memory?",
        retrieved_chunks=[],
    )

    # When
    result = await grader.grade(state)

    # Then
    assert result.is_relevant is False
    assert result.reasoning.strip() != ""


async def test_grader_flags_off_topic_chunks_as_insufficient(
    ollama_llm: OllamaLLM,
    retrieval_service: RetrievalService,
) -> None:
    """Verify grader rejects chunks from arXiv:2005.11401 for an unrelated query."""
    # Given
    query_text = "How does Kubernetes horizontal pod autoscaling calculate replica counts?"
    chunks = await retrieval_service.retrieve(
        Query(text=query_text),
        initial_top_k=30,
        final_top_k=6,
    )
    assert chunks
    grader = RelevanceGrader(ollama_llm)
    state = AgentState(
        original_query=query_text,
        current_query=query_text,
        retrieved_chunks=chunks,
    )

    # When
    result = await grader.grade(state)

    # Then
    assert result.is_relevant is False
    assert result.reasoning.strip() != ""


async def test_grader_flags_on_topic_chunks_as_sufficient(
    ollama_llm: OllamaLLM,
    retrieval_service: RetrievalService,
) -> None:
    """Verify grader accepts clearly on-topic chunks retrieved from arXiv:2005.11401."""
    # Given
    query_text = (
        "What is the difference between parametric memory and non-parametric memory in RAG?"
    )
    chunks = await retrieval_service.retrieve(
        Query(text=query_text),
        initial_top_k=30,
        final_top_k=6,
    )
    assert chunks
    grader = RelevanceGrader(ollama_llm)
    state = AgentState(
        original_query=query_text,
        current_query=query_text,
        retrieved_chunks=chunks,
    )

    # When
    result = await grader.grade(state)

    # Then
    assert result.is_relevant is True
    assert result.reasoning.strip() != ""
