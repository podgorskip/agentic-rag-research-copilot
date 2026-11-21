import pytest

from agentic_rag_copilot.agent import QueryRewriter
from agentic_rag_copilot.domain.exceptions import LLMError
from agentic_rag_copilot.domain.models import AgentState
from tests.mocks.agent import MockLLM


async def test_rewrite_produces_new_query_and_includes_history_and_reasoning() -> None:
    """Verify rewriter trims output, preserves state, and sends history and reasoning to LLM."""
    # Given
    llm = MockLLM(text_response="  FAISS HNSW Maximum Inner Product Search Wikipedia index  ")
    rewriter = QueryRewriter(llm)
    state = AgentState(
        original_query="How does RAG index Wikipedia for fast retrieval?",
        current_query="RAG Wikipedia retrieval index",
        query_history=[
            "How does RAG index Wikipedia for fast retrieval?",
            "RAG Wikipedia retrieval index",
        ],
    )
    reasoning = "Chunks mentioned Wikipedia but did not describe the MIPS or FAISS index structure."

    # When
    rewritten = await rewriter.rewrite(state, reasoning=reasoning)

    # Then
    assert rewritten == "FAISS HNSW Maximum Inner Product Search Wikipedia index"
    assert state.current_query == "RAG Wikipedia retrieval index"
    assert state.query_history == [
        "How does RAG index Wikipedia for fast retrieval?",
        "RAG Wikipedia retrieval index",
    ]
    assert len(llm.text_calls) == 1
    prompt, system_prompt = llm.text_calls[0]
    assert system_prompt is not None
    assert "do not repeat" in system_prompt.lower()
    assert state.original_query in prompt
    assert state.current_query in prompt
    assert "RAG Wikipedia retrieval index" in prompt
    assert reasoning in prompt


async def test_rewrite_handles_empty_history_and_no_reasoning() -> None:
    """Verify rewriter handles empty history, omitted reasoning, and wrapping quotes."""
    # Given
    llm = MockLLM(text_response='"RAG-Token per-token document marginalization"')
    rewriter = QueryRewriter(llm)
    state = AgentState(
        original_query="How does RAG-Token work?",
        current_query="How does RAG-Token work?",
        query_history=[],
    )

    # When
    rewritten = await rewriter.rewrite(state)

    # Then
    assert rewritten == "RAG-Token per-token document marginalization"
    assert len(llm.text_calls) == 1
    prompt, _ = llm.text_calls[0]
    assert "None" in prompt
    assert "Not provided." in prompt


async def test_rewrite_raises_value_error_on_empty_llm_output() -> None:
    """Verify rewriter raises ``ValueError`` when the LLM returns whitespace-only text."""
    # Given
    llm = MockLLM(text_response="   \n\t  ")
    rewriter = QueryRewriter(llm)
    state = AgentState(
        original_query="What is FEVER?",
        current_query="What is FEVER?",
    )

    # When / Then
    with pytest.raises(ValueError, match="empty rewritten query"):
        await rewriter.rewrite(state, reasoning="Context lacked FEVER results.")


async def test_rewrite_propagates_llm_error() -> None:
    """Verify rewriter propagates ``LLMError`` from the underlying LLM adapter."""
    # Given
    llm = MockLLM(fail_with=LLMError("Ollama timeout"))
    rewriter = QueryRewriter(llm)
    state = AgentState(
        original_query="What is FEVER?",
        current_query="What is FEVER?",
    )

    # When / Then
    with pytest.raises(LLMError, match="Ollama timeout"):
        await rewriter.rewrite(state, reasoning="Missing FEVER scores.")
