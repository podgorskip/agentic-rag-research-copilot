from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from agentic_rag_copilot.domain.exceptions import LLMError
from agentic_rag_copilot.domain.protocols import LLMProtocol

T = TypeVar("T", bound=BaseModel)

DEFAULT_LLM_MODEL = "llama3.2"


class _GenerateResponse(BaseModel):
    response: str


class OllamaLLM(LLMProtocol):
    """Generates text and structured output through the Ollama ``/api/generate`` endpoint."""

    def __init__(self, client: httpx.AsyncClient, model: str = DEFAULT_LLM_MODEL) -> None:
        """Create the LLM adapter.

        Args:
            client: HTTP client configured with the Ollama base URL. The caller owns its lifecycle.
            model: Name of a generation model already pulled into Ollama.
        """
        self._client = client
        self._model = model

    async def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a raw string response from the LLM.

        Args:
            prompt: Text prompt sent to the LLM.
            system_prompt: Optional system instruction guiding model behavior.

        Returns:
            Generated string response.

        Raises:
            LLMError: If the request fails or the response is unusable.
        """
        return await self._generate(prompt, system_prompt=system_prompt, format_schema=None)

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

        Raises:
            LLMError: If the request fails or the response does not match the schema.
        """
        raw = await self._generate(
            prompt,
            system_prompt=system_prompt,
            format_schema=response_model.model_json_schema(),
        )
        try:
            return response_model.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMError(
                f"Ollama response did not match schema {response_model.__name__}: {raw!r}"
            ) from exc

    async def _generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        format_schema: dict[str, object] | None,
    ) -> str:
        payload: dict[str, object] = {"model": self._model, "prompt": prompt, "stream": False}
        if system_prompt is not None:
            payload["system"] = system_prompt
        if format_schema is not None:
            payload["format"] = format_schema

        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc!r}") from exc

        try:
            return _GenerateResponse.model_validate_json(response.content).response
        except ValidationError as exc:
            raise LLMError("Ollama returned a malformed generation response") from exc
