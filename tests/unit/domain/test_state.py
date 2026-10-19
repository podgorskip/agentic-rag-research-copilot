import pytest
from pydantic import ValidationError

from agentic_rag_copilot.domain.models import (
    AgentState,
    Chunk,
    Citation,
    GradeResult,
    SynthesizedAnswer,
)


def test_grade_result_immutability() -> None:
    grade = GradeResult(is_relevant=True, reasoning="Contains exact metrics.")
    assert grade.is_relevant is True

    with pytest.raises(ValidationError):
        grade.is_relevant = False


def test_agent_state_mutability_and_progression() -> None:
    state = AgentState(
        original_query="What is ColPali?",
        current_query="What is ColPali?",
    )
    assert state.loop_count == 0
    assert state.is_context_sufficient is False
    assert state.final_answer is None

    state.loop_count += 1
    state.current_query = "ColPali multimodal retrieval architecture"
    state.query_history.append("What is ColPali?")

    chunk = Chunk(id="c1", document_id="d1", text="ColPali uses vision language models.")
    state.retrieved_chunks = [chunk]
    state.is_context_sufficient = True

    citation = Citation(
        chunk_id="c1",
        document_id="d1",
        source_uri="file://paper.pdf",
        quote="ColPali uses vision language models.",
    )
    state.final_answer = SynthesizedAnswer(
        answer="ColPali is a multimodal retrieval model [1].",
        citations=[citation],
    )

    assert state.loop_count == 1
    assert len(state.retrieved_chunks) == 1
    assert state.final_answer.citations[0].chunk_id == "c1"


def test_agent_state_rejects_negative_loop_count() -> None:
    with pytest.raises(ValidationError):
        AgentState(
            original_query="test",
            current_query="test",
            loop_count=-1,
        )
