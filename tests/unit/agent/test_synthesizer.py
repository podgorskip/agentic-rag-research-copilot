import pytest

from agentic_rag_copilot.agent import AnswerSynthesizer
from agentic_rag_copilot.domain.exceptions import LLMError
from agentic_rag_copilot.domain.models import AgentState, Chunk, Citation, SynthesizedAnswer
from tests.mocks.agent import MockLLM


def _make_chunk(chunk_id: str, text: str, source_uri: str | None = "paper.pdf") -> Chunk:
    """Build a ``Chunk`` fixture for synthesizer unit tests.

    Args:
        chunk_id: Unique chunk identifier.
        text: Chunk text content.
        source_uri: Value stored under ``metadata["source_uri"]``; omitted when ``None``.

    Returns:
        A ``Chunk`` attached to ``doc-1``.
    """
    metadata = {} if source_uri is None else {"source_uri": source_uri}
    return Chunk(id=chunk_id, document_id="doc-1", text=text, metadata=metadata)


def _state(chunks: list[Chunk]) -> AgentState:
    return AgentState(
        original_query="Which retriever does RAG use?",
        current_query="RAG retriever DPR bi-encoder",
        retrieved_chunks=chunks,
    )


async def test_synthesize_returns_grounded_answer_and_sends_labelled_chunks() -> None:
    """Verify happy path: citations are kept, grounded identifiers overwrite LLM values."""
    # Given
    llm = MockLLM(
        structured_response=SynthesizedAnswer(
            answer="RAG uses a DPR bi-encoder retriever [1].",
            citations=[
                Citation(
                    chunk_id="c1",
                    document_id="hallucinated-doc",
                    source_uri="https://made.up/source",
                    quote="DPR bi-encoder",
                )
            ],
        )
    )
    synthesizer = AnswerSynthesizer(llm)
    state = _state(
        [
            _make_chunk("c1", "The retriever is a DPR bi-encoder.", source_uri="rag.pdf"),
            _make_chunk("c2", "The generator is BART-large."),
        ]
    )

    # When
    result = await synthesizer.synthesize(state)

    # Then
    assert result.answer == "RAG uses a DPR bi-encoder retriever [1]."
    assert result.citations == [
        Citation(chunk_id="c1", document_id="doc-1", source_uri="rag.pdf", quote="DPR bi-encoder")
    ]
    assert state.final_answer is None
    assert len(llm.structured_calls) == 1
    prompt, response_model, system_prompt = llm.structured_calls[0]
    assert response_model is SynthesizedAnswer
    assert system_prompt is not None
    assert "[1]" in system_prompt
    assert state.original_query in prompt
    assert "[1] chunk_id=c1" in prompt
    assert "[2] chunk_id=c2" in prompt
    assert "The generator is BART-large." in prompt


async def test_synthesize_falls_back_to_document_id_when_source_uri_missing() -> None:
    """Verify citations use ``document_id`` as ``source_uri`` when chunk metadata lacks it."""
    # Given
    llm = MockLLM(
        structured_response=SynthesizedAnswer(
            answer="BART [1].",
            citations=[Citation(chunk_id="c1", document_id="x", source_uri="y", quote="BART")],
        )
    )
    state = _state([_make_chunk("c1", "BART generator.", source_uri=None)])

    # When
    result = await AnswerSynthesizer(llm).synthesize(state)

    # Then
    assert result.citations[0].source_uri == "doc-1"


async def test_synthesize_rejects_citation_of_unknown_chunk() -> None:
    """Verify a citation whose ``chunk_id`` was never retrieved raises ``ValueError``."""
    # Given
    llm = MockLLM(
        structured_response=SynthesizedAnswer(
            answer="RAG uses DPR [1] and FAISS [2].",
            citations=[
                Citation(chunk_id="c1", document_id="doc-1", source_uri="p", quote="DPR"),
                Citation(chunk_id="ghost", document_id="doc-1", source_uri="p", quote="FAISS"),
            ],
        )
    )
    state = _state([_make_chunk("c1", "DPR retriever.")])

    # When / Then
    with pytest.raises(ValueError, match="unknown chunk_id 'ghost'"):
        await AnswerSynthesizer(llm).synthesize(state)


async def test_synthesize_short_circuits_without_calling_llm_when_chunks_empty() -> None:
    """Verify empty context yields a no-information answer without an LLM call."""
    # Given
    llm = MockLLM(structured_response=SynthesizedAnswer(answer="should not be used"))

    # When
    result = await AnswerSynthesizer(llm).synthesize(_state([]))

    # Then
    assert result.answer.strip() != ""
    assert result.citations == []
    assert llm.structured_calls == []


async def test_synthesize_propagates_llm_error() -> None:
    """Verify ``LLMError`` from the adapter propagates unchanged."""
    # Given
    llm = MockLLM(fail_with=LLMError("Ollama unreachable"))

    # When / Then
    with pytest.raises(LLMError, match="Ollama unreachable"):
        await AnswerSynthesizer(llm).synthesize(_state([_make_chunk("c1", "text")]))
