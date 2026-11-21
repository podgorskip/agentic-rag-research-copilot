import pytest

from agentic_rag_copilot.agent import RelevanceGrader
from agentic_rag_copilot.domain.exceptions import LLMError
from agentic_rag_copilot.domain.models import AgentState, Chunk, GradeResult
from tests.mocks.agent import MockLLM


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    """Build a minimal ``Chunk`` fixture for grader unit tests.

    Args:
        chunk_id: Unique chunk identifier.
        text: Chunk text content.

    Returns:
        A ``Chunk`` attached to ``doc-1``.
    """
    return Chunk(id=chunk_id, document_id="doc-1", text=text)


async def test_grade_returns_relevant_result_for_sufficient_context() -> None:
    """Verify grader returns relevant ``GradeResult`` and includes query and chunks in prompt."""
    # Given
    expected = GradeResult(
        is_relevant=True,
        reasoning="Chunk 1 directly explains RAG-Sequence marginalization.",
    )
    llm = MockLLM(structured_response=expected)
    grader = RelevanceGrader(llm)
    state = AgentState(
        original_query="How does RAG-Sequence marginalize over documents?",
        current_query="RAG-Sequence document marginalization",
        retrieved_chunks=[
            _make_chunk("c1", "RAG-Sequence uses the same retrieved document for the sequence."),
            _make_chunk("c2", "Marginalization sums over the top-K retrieved documents."),
        ],
    )

    # When
    result = await grader.grade(state)

    # Then
    assert result == expected
    assert state.is_context_sufficient is False
    assert len(llm.structured_calls) == 1
    prompt, response_model, system_prompt = llm.structured_calls[0]
    assert response_model is GradeResult
    assert system_prompt is not None
    assert state.current_query in prompt
    assert "RAG-Sequence uses the same retrieved document" in prompt
    assert "Marginalization sums over the top-K" in prompt


async def test_grade_returns_irrelevant_result_for_insufficient_context() -> None:
    """Verify grader returns irrelevant ``GradeResult`` without mutating ``AgentState``."""
    # Given
    expected = GradeResult(
        is_relevant=False,
        reasoning="Retrieved chunks discuss unrelated citation formatting.",
    )
    llm = MockLLM(structured_response=expected)
    grader = RelevanceGrader(llm)
    state = AgentState(
        original_query="What is FAISS HNSW?",
        current_query="FAISS HNSW index",
        retrieved_chunks=[_make_chunk("c1", "Broader Impact discusses societal considerations.")],
    )

    # When
    result = await grader.grade(state)

    # Then
    assert result == expected
    assert result.is_relevant is False
    assert state.is_context_sufficient is False
    assert len(llm.structured_calls) == 1


async def test_grade_short_circuits_without_calling_llm_when_chunks_empty() -> None:
    """Verify grader short-circuits to ``is_relevant=False`` without calling the LLM."""
    # Given
    llm = MockLLM(
        structured_response=GradeResult(is_relevant=True, reasoning="Should never be returned.")
    )
    grader = RelevanceGrader(llm)
    state = AgentState(
        original_query="What is DPR?",
        current_query="Dense Passage Retriever",
        retrieved_chunks=[],
    )

    # When
    result = await grader.grade(state)

    # Then
    assert result.is_relevant is False
    assert result.reasoning.strip() != ""
    assert llm.structured_calls == []
    assert state.is_context_sufficient is False


async def test_grade_propagates_llm_error() -> None:
    """Verify grader propagates ``LLMError`` from the underlying LLM adapter."""
    # Given
    llm = MockLLM(fail_with=LLMError("Ollama unreachable"))
    grader = RelevanceGrader(llm)
    state = AgentState(
        original_query="What is DPR?",
        current_query="What is DPR?",
        retrieved_chunks=[_make_chunk("c1", "DPR uses a BERT-base bi-encoder.")],
    )

    # When / Then
    with pytest.raises(LLMError, match="Ollama unreachable"):
        await grader.grade(state)
