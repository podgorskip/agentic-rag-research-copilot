from typing import Final

from agentic_rag_copilot.domain.models import AgentState, GradeResult
from agentic_rag_copilot.domain.protocols import LLMProtocol
from agentic_rag_copilot.logging import get_logger

_logger = get_logger(__name__)

_EMPTY_CHUNKS_REASONING: Final[str] = "No chunks were retrieved to answer the query."

_SYSTEM_PROMPT: Final[str] = (
    "You are a strict relevance grader for a retrieval-augmented generation system. "
    "Evaluate whether the provided context chunks contain enough accurate information "
    "to answer the user's query. Return is_relevant=true only when the context is "
    "sufficient to answer the query; otherwise return is_relevant=false. "
    "Always provide a concise, non-empty explanation in reasoning stating what the "
    "context covers or what information is missing."
)


class RelevanceGrader:
    """Grades whether retrieved chunks are sufficient to answer the current query.

    Caller-side contract:
        ``grade`` is a pure function of ``state`` and does not mutate ``AgentState``.
        The orchestrating loop is responsible for writing ``result.is_relevant`` back
        to ``state.is_context_sufficient``.
    """

    def __init__(self, llm: LLMProtocol) -> None:
        """Initialize the relevance grader.

        Args:
            llm: LLM adapter implementing ``LLMProtocol`` for structured grading.
        """
        self._llm = llm

    async def grade(self, state: AgentState) -> GradeResult:
        """Grade whether ``state.retrieved_chunks`` suffice to answer ``state.current_query``.

        Short-circuits without calling the LLM when ``state.retrieved_chunks`` is empty.
        Does not mutate ``state``; callers must assign ``state.is_context_sufficient``
        from the returned ``GradeResult``.

        Args:
            state: Current agent state containing ``current_query`` and ``retrieved_chunks``.

        Returns:
            Structured ``GradeResult`` with ``is_relevant`` and ``reasoning``.

        Raises:
            LLMError: Propagated if the underlying LLM call fails or returns invalid schema.
        """
        if not state.retrieved_chunks:
            _logger.debug("Short-circuiting relevance grading: no retrieved chunks.")
            return GradeResult(is_relevant=False, reasoning=_EMPTY_CHUNKS_REASONING)

        prompt = _build_prompt(state)
        _logger.debug(
            f"Grading {len(state.retrieved_chunks)} chunk(s) for query: {state.current_query!r}"
        )
        return await self._llm.generate_structured(
            prompt,
            GradeResult,
            system_prompt=_SYSTEM_PROMPT,
        )


def _build_prompt(state: AgentState) -> str:
    """Build the grading prompt from the current query and retrieved chunks.

    Args:
        state: Agent state providing ``current_query`` and ``retrieved_chunks``.

    Returns:
        Formatted prompt containing the query and numbered chunk excerpts.
    """
    formatted_chunks = "\n\n".join(
        f"[{idx}] {chunk.text}" for idx, chunk in enumerate(state.retrieved_chunks, start=1)
    )
    return f"Query:\n{state.current_query}\n\nRetrieved Context:\n{formatted_chunks}"
