from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMProtocol(Protocol):
    """Contract for Ollama and external LLM APIs."""

    async def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a raw string response from the LLM.

        Args:
            prompt: Text prompt sent to the LLM.
            system_prompt: Optional system instruction guiding model behavior.

        Returns:
            Generated string response.
        """
        ...

    async def generate_structured(
        self, prompt: str, response_model: type[T], system_prompt: str | None = None
    ) -> T:
        """Generate structured output adhering to a specific Pydantic schema.

        Args:
            prompt: Text prompt sent to the LLM.
            response_model: Pydantic model class defining the expected output structure.
            system_prompt: Optional system instruction guiding model behavior.

        Returns:
            Validated instance of the requested response model.
        """
        ...
