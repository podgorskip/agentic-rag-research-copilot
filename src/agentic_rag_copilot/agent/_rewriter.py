from typing import Final

from agentic_rag_copilot.domain.models import AgentState
from agentic_rag_copilot.domain.protocols import LLMProtocol
from agentic_rag_copilot.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT: Final[str] = (
    "You are a search query reformulation assistant for a technical RAG system. "
    "Given the original user question, the current search query, previously attempted "
    "queries, and why the retrieved context was insufficient, write a single improved "
    "search query that is more specific and likely to retrieve the missing information. "
    "Do NOT repeat any query already listed in the query history. "
    "Return only the rewritten query text with no explanation or surrounding commentary."
)


class QueryRewriter:
    """Reformulates search queries when retrieved context is insufficient.

    Caller-side contract:
        ``rewrite`` does not mutate ``AgentState``. The orchestrating loop is
        responsible for appending the previous or new query to ``state.query_history``
        and updating ``state.current_query``.
    """

    def __init__(self, llm: LLMProtocol) -> None:
        """Initialize the query rewriter.

        Args:
            llm: LLM adapter implementing ``LLMProtocol`` for text generation.
        """
        self._llm = llm

    async def rewrite(self, state: AgentState, reasoning: str | None = None) -> str:
        """Generate a reformulated search query based on state and grader feedback.

        Does not mutate ``state``; callers own updating ``state.current_query`` and
        ``state.query_history``.

        Args:
            state: Current agent state containing ``original_query``, ``current_query``,
                and ``query_history``.
            reasoning: Optional explanation from ``GradeResult.reasoning`` describing
                why the previously retrieved context was insufficient.

        Returns:
            Non-empty rewritten query string.

        Raises:
            ValueError: If the LLM returns an empty or whitespace-only query.
            LLMError: Propagated if the underlying LLM call fails.
        """
        prompt = _build_prompt(state, reasoning)
        logger.debug(
            f"Rewriting query {state.current_query!r} (history_len={len(state.query_history)})"
        )
        raw_rewrite = await self._llm.generate_text(prompt, system_prompt=_SYSTEM_PROMPT)
        rewritten = _clean_query(raw_rewrite)
        if not rewritten:
            raise ValueError("QueryRewriter received an empty rewritten query from the LLM.")
        return rewritten


def _build_prompt(state: AgentState, reasoning: str | None) -> str:
    """Build the query-rewriting prompt from state history and optional grader reasoning.

    Args:
        state: Agent state providing query context and history.
        reasoning: Optional grader explanation of why retrieved context was lacking.

    Returns:
        Formatted prompt string for the LLM.
    """
    if state.query_history:
        history_block = "\n".join(f"- {query}" for query in state.query_history)
    else:
        history_block = "None"

    reasoning_text = reasoning.strip() if reasoning and reasoning.strip() else "Not provided."

    return (
        f"Original Question:\n{state.original_query}\n\n"
        f"Current Query:\n{state.current_query}\n\n"
        f"Prior Queries (do NOT repeat any of these):\n{history_block}\n\n"
        f"Why Retrieved Context Was Insufficient:\n{reasoning_text}\n\n"
        "Rewritten Query:"
    )


def _clean_query(raw_text: str) -> str:
    """Normalize LLM rewrite output by stripping whitespace and wrapping quotes.

    Args:
        raw_text: Raw text returned by the LLM.

    Returns:
        Cleaned single-query string (may be empty if the input was blank).
    """
    cleaned = raw_text.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned
