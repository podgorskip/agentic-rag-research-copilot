"""``POST /ask`` and ``POST /ask/stream``: answer questions via the agentic RAG loop."""

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Final

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from agentic_rag_copilot.agent import AgenticLoopService
from agentic_rag_copilot.domain.models import AskRequest, AskResponse
from agentic_rag_copilot.logging import get_logger

from ._dependencies import AgenticLoopServiceDep

_logger = get_logger(__name__)

router = APIRouter()

_SYNTHESIS_ERROR_DETAIL: Final[str] = "Unable to produce a grounded answer for this query"
_SSE_MEDIA_TYPE: Final[str] = "text/event-stream"
_SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

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

_ASK_STREAM_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    200: {
        "description": (
            "Server-Sent Events stream of ``AgentEvent`` frames "
            "(``event: <type>\\ndata: <json>\\n\\n``), terminated by ``event: done`` "
            "after ``answer`` or closed immediately after an ``error`` event."
        ),
        "content": {
            _SSE_MEDIA_TYPE: {
                "schema": {"type": "string"},
                "example": (
                    'event: retrieve\ndata: {"type":"retrieve","iteration":1,"payload":{}}\n\n'
                    'event: answer\ndata: {"type":"answer","iteration":1,"payload":{}}\n\n'
                    "event: done\ndata: {}\n\n"
                ),
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


@router.post(
    "/ask/stream",
    status_code=200,
    response_class=StreamingResponse,
    summary="Stream agentic RAG progress and answer over SSE",
    tags=["Ask"],
    responses=_ASK_STREAM_RESPONSES,
)
async def ask_stream(
    payload: AskRequest,
    request: Request,
    loop: AgenticLoopServiceDep,
) -> StreamingResponse:
    """Stream agentic loop progress and the final answer as Server-Sent Events.

    Each ``AgentEvent`` is formatted as ``event: <type>\\ndata: <json>\\n\\n``, followed by
    ``event: done\\ndata: {}\\n\\n`` after the terminal ``answer`` event. If an ``error`` event
    occurs mid-stream, it is forwarded and the stream closes immediately. Client disconnects
    close the underlying async generator to avoid orphaned LLM calls.

    Args:
        payload: Validated question, retrieval ``top_k``, and optional metadata filters.
        request: Raw FastAPI request used to detect client disconnects.
        loop: Injected ``AgenticLoopService`` producing ``AgentEvent`` items.

    Returns:
        ``StreamingResponse`` with ``text/event-stream`` media type and anti-buffering headers.
    """
    return StreamingResponse(
        _stream_sse_events(request, payload, loop),
        media_type=_SSE_MEDIA_TYPE,
        headers=_SSE_HEADERS,
    )


async def _stream_sse_events(
    request: Request,
    payload: AskRequest,
    loop: AgenticLoopService,
) -> AsyncIterator[str]:
    """Convert ``loop.ask_stream`` events into SSE frames and close the stream on exit.

    Args:
        request: Raw request checked for disconnects between events.
        payload: Validated ``AskRequest``.
        loop: Agentic loop service yielding ``AgentEvent`` objects.

    Yields:
        SSE-formatted event frames.
    """
    stream = loop.ask_stream(payload)
    try:
        async for event in stream:
            if await request.is_disconnected():
                _logger.info("Client disconnected during POST /ask/stream; cancelling stream.")
                return
            yield _format_sse(event.type, event.model_dump_json())
            if event.type == "error":
                return
            if event.type == "answer":
                yield _format_sse("done", "{}")
                return
    finally:
        if isinstance(stream, AsyncGenerator):
            await stream.aclose()


def _format_sse(event_type: str, data: str) -> str:
    """Format a single Server-Sent Events frame.

    Args:
        event_type: SSE ``event:`` name.
        data: Single-line JSON payload for ``data:``.

    Returns:
        RFC-compliant SSE frame terminated by a blank line.
    """
    return f"event: {event_type}\ndata: {data}\n\n"
