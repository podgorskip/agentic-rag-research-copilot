"""Exercises OllamaLLM against a real local Ollama instance.

Skipped automatically when Ollama is unreachable at the configured RAG_OLLAMA_URL (default
http://localhost:11434), so the suite stays green on machines without it installed. Run with
Ollama up and the configured RAG_LLM_MODEL pulled (see README) to exercise it for real.
"""

from collections.abc import AsyncIterator

import httpx
import pytest

from agentic_rag_copilot.adapters.llm import LLMError, OllamaLLM
from agentic_rag_copilot.agent import AgentSettings
from agentic_rag_copilot.domain.models import GradeResult

_SETTINGS = AgentSettings()


def _ollama_is_reachable() -> bool:
    try:
        httpx.get(f"{_SETTINGS.ollama_url}/api/tags", timeout=2.0).raise_for_status()
    except httpx.HTTPError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _ollama_is_reachable(),
    reason=f"Ollama is not reachable at {_SETTINGS.ollama_url}; start it to run this test",
)


@pytest.fixture
async def llm() -> AsyncIterator[OllamaLLM]:
    async with httpx.AsyncClient(
        base_url=_SETTINGS.ollama_url, timeout=_SETTINGS.llm_timeout_seconds
    ) as client:
        yield OllamaLLM(client, model=_SETTINGS.llm_model)


async def test_generate_text_returns_a_real_completion(llm: OllamaLLM) -> None:
    # When
    response = await llm.generate_text(
        "Reply with a single word that means 'yes'.", system_prompt="Answer in one word."
    )

    # Then
    assert isinstance(response, str)
    assert response.strip() != ""


async def test_generate_structured_returns_a_schema_conformant_model(llm: OllamaLLM) -> None:
    # Given
    prompt = (
        "The retrieved context is empty. Grade whether it is relevant to the question "
        "'What is Reciprocal Rank Fusion?' and explain your reasoning."
    )

    # When
    result = await llm.generate_structured(prompt, GradeResult)

    # Then
    assert isinstance(result, GradeResult)
    assert isinstance(result.is_relevant, bool)
    assert result.reasoning.strip() != ""


async def test_connection_failure_raises_llm_error() -> None:
    # Given
    async with httpx.AsyncClient(base_url="http://127.0.0.1:1", timeout=1.0) as client:
        unreachable_llm = OllamaLLM(client, model=_SETTINGS.llm_model)

        # When / Then
        with pytest.raises(LLMError):
            await unreachable_llm.generate_text("hello")
