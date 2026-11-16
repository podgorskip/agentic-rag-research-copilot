import pytest

from agentic_rag_copilot.domain.models import GradeResult
from agentic_rag_copilot.domain.protocols import LLMProtocol
from tests.mocks.agent import MockLLM


def test_mock_llm_satisfies_protocol() -> None:
    # When
    llm = MockLLM()

    # Then
    assert isinstance(llm, LLMProtocol)


async def test_generate_text_returns_canned_response_and_records_call() -> None:
    # Given
    llm = MockLLM(text_response="hello")

    # When
    result = await llm.generate_text("prompt", system_prompt="sys")

    # Then
    assert result == "hello"
    assert llm.text_calls == [("prompt", "sys")]


async def test_generate_structured_returns_canned_response_and_records_call() -> None:
    # Given
    grade = GradeResult(is_relevant=True, reasoning="Sufficient context.")
    llm = MockLLM(structured_response=grade)

    # When
    result = await llm.generate_structured("prompt", GradeResult)

    # Then
    assert result == grade
    assert llm.structured_calls == [("prompt", GradeResult, None)]


async def test_generate_structured_raises_without_configured_response() -> None:
    # Given
    llm = MockLLM()

    # When / Then
    with pytest.raises(ValueError, match="no configured structured_response"):
        await llm.generate_structured("prompt", GradeResult)


async def test_generate_text_raises_configured_failure() -> None:
    # Given
    llm = MockLLM(fail_with=ConnectionError("down"))

    # When / Then
    with pytest.raises(ConnectionError):
        await llm.generate_text("prompt")


async def test_generate_structured_raises_configured_failure() -> None:
    # Given
    llm = MockLLM(fail_with=ConnectionError("down"))

    # When / Then
    with pytest.raises(ConnectionError):
        await llm.generate_structured("prompt", GradeResult)
