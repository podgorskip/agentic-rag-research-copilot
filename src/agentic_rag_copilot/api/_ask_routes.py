"""``POST /ask``: answer questions via the self-correcting agentic RAG loop."""

from typing import Any, Final

from fastapi import APIRouter, HTTPException

from agentic_rag_copilot.domain.models import AskRequest, AskResponse
from agentic_rag_copilot.logging import get_logger

from ._dependencies import AgenticLoopServiceDep

_logger = get_logger(__name__)

router = APIRouter()

_SYNTHESIS_ERROR_DETAIL: Final[str] = "Unable to produce a grounded answer for this query"

_ASK_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    502: {
        "description": "The LLM produced an ungrounded or invalid answer.",
        "content": {
            "application/json": {
                "example": {"detail": _SYNTHESIS_ERROR_DETAIL},
            }
        },
    },
    503: {
        "description": "A backing service (LLM, embedder, or vector store) is unavailable.",
        "content": {
            "application/json": {
                "example": {"detail": "A backing service is unavailable; try again later"},
            }
        },
    },
}


@router.post(
    "/ask",
    status_code=200,
    response_model=AskResponse,
    summary="Ask a question over ingested documents",
    tags=["Ask"],
    responses=_ASK_RESPONSES,
)
async def ask(payload: AskRequest, loop: AgenticLoopServiceDep) -> AskResponse:
    """Run the agentic RAG loop and return the synthesized answer with grounded citations.

    Args:
        payload: Validated question, retrieval ``top_k``, and optional metadata filters.
        loop: Injected ``AgenticLoopService`` orchestrating retrieval, grading, and synthesis.

    Returns:
        ``AskResponse`` containing the original query, cited answer, citations, and iteration
        count exactly as produced by ``AgenticLoopService``.

    Raises:
        HTTPException: 502 if the synthesizer rejects an ungrounded citation or the rewriter
            produces an empty query.
        LLMError: Propagated to the app-level 503 handler on LLM backend failures.
        EmbeddingError: Propagated to the app-level 503 handler on embedding backend failures.
    """
    try:
        return await loop.ask(payload)
    except ValueError as exc:
        _logger.error(f"POST /ask failed synthesis or rewrite validation: {exc}")
        raise HTTPException(status_code=502, detail=_SYNTHESIS_ERROR_DETAIL) from exc
