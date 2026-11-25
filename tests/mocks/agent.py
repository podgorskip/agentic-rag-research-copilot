"""Deterministic LLM fake, for use in the test suite only.

Records how it was called and can be told to fail, so a test can exercise both the happy path
and error handling without a running Ollama instance.
"""

import re
from typing import Final, TypeVar

from pydantic import BaseModel

from agentic_rag_copilot.domain.models import Citation, GradeResult, SynthesizedAnswer
from agentic_rag_copilot.domain.protocols import LLMProtocol

T = TypeVar("T", bound=BaseModel)

_CHUNK_ID_RE: Final[re.Pattern[str]] = re.compile(r"chunk_id=([^\s\]]+)")


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


class ScriptedLoopLLM(LLMProtocol):
    """Deterministic LLM fake for multi-stage loop tests (grading, rewriting, synthesis).

    Returns preset boolean verdicts for ``GradeResult``, incremental rewritten queries for
    ``generate_text``, and a ``SynthesizedAnswer`` whose citation grounds itself in the first
    ``chunk_id=...`` label present in the prompt (or ``hallucinate_chunk_id`` when set).
    """

    def __init__(
        self,
        grade_verdicts: list[bool] | None = None,
        *,
        fail_with: Exception | None = None,
        hallucinate_chunk_id: str | None = None,
    ) -> None:
        """Initialize the scripted loop LLM.

        Args:
            grade_verdicts: Sequence of ``is_relevant`` verdicts for successive grade calls.
            fail_with: Optional exception raised on every LLM call.
            hallucinate_chunk_id: Optional unknown chunk ID to emit in ``SynthesizedAnswer``.
        """
        self._verdicts = iter(grade_verdicts or [True])
        self._fail_with = fail_with
        self._hallucinate_chunk_id = hallucinate_chunk_id
        self._rewrite_index = 0

    async def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """Return a deterministic rewritten query or raise ``_fail_with``.

        Args:
            prompt: Rewrite prompt text.
            system_prompt: Optional system prompt.

        Returns:
            Non-empty rewritten query string.
        """
        if self._fail_with is not None:
            raise self._fail_with
        self._rewrite_index += 1
        return f"refined retrieval query pass {self._rewrite_index}"

    async def generate_structured(
        self, prompt: str, response_model: type[T], system_prompt: str | None = None
    ) -> T:
        """Return a scripted ``GradeResult`` or grounded ``SynthesizedAnswer``.

        Args:
            prompt: Grading or synthesis prompt containing retrieved chunk labels.
            response_model: Expected Pydantic output model.
            system_prompt: Optional system prompt.

        Returns:
            Validated instance of ``response_model``.
        """
        if self._fail_with is not None:
            raise self._fail_with
        if response_model is GradeResult:
            verdict = next(self._verdicts, True)
            payload: BaseModel = GradeResult(is_relevant=verdict, reasoning="scripted")
        else:
            match = _CHUNK_ID_RE.search(prompt)
            chunk_id = self._hallucinate_chunk_id or (match.group(1) if match else "missing-id")
            payload = SynthesizedAnswer(
                answer="Grounded summary from the paper [1].",
                citations=[
                    Citation(
                        chunk_id=chunk_id,
                        document_id="placeholder-doc",
                        source_uri="placeholder-uri",
                        quote="supporting excerpt",
                    )
                ],
            )
        return response_model.model_validate(payload.model_dump())
