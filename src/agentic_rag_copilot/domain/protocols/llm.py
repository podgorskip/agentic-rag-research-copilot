from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMProtocol(Protocol):
    """Contract for Ollama and external LLM APIs."""

    async def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generates a raw string response (e.g., for simple query rewriting)."""
        ...

    async def generate_structured(
        self, prompt: str, response_model: type[T], system_prompt: str | None = None
    ) -> T:
        """Forces the LLM to return output adhering to a specific Pydantic schema."""
        ...
