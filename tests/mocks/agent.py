"""Deterministic LLM fake, for use in the test suite only.

Records how it was called and can be told to fail, so a test can exercise both the happy path
and error handling without a running Ollama instance.
"""

from typing import TypeVar

from pydantic import BaseModel

from agentic_rag_copilot.domain.protocols import LLMProtocol

T = TypeVar("T", bound=BaseModel)


class MockLLM(LLMProtocol):
    """Returns a configured canned response from generate_text / generate_structured."""

    def __init__(
        self,
        text_response: str = "mock response",
        structured_response: BaseModel | None = None,
        *,
        fail_with: Exception | None = None,
    ) -> None:
        self.text_response = text_response
        self.structured_response = structured_response
        self.fail_with = fail_with
        self.text_calls: list[tuple[str, str | None]] = []
        self.structured_calls: list[tuple[str, type[BaseModel], str | None]] = []

    async def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        self.text_calls.append((prompt, system_prompt))
        if self.fail_with is not None:
            raise self.fail_with
        return self.text_response

    async def generate_structured(
        self, prompt: str, response_model: type[T], system_prompt: str | None = None
    ) -> T:
        self.structured_calls.append((prompt, response_model, system_prompt))
        if self.fail_with is not None:
            raise self.fail_with
        if self.structured_response is None:
            raise ValueError("MockLLM has no configured structured_response")
        return response_model.model_validate(self.structured_response.model_dump())
