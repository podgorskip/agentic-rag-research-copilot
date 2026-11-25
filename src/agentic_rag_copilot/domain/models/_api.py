from typing import Any, Literal

from pydantic import Field

from ._base import ImmutableModel
from ._entities import Citation, SourceType

AgentEventType = Literal["rewrite", "retrieve", "grade", "generate", "answer", "error"]


class IngestRequest(ImmutableModel):
    source_uri: str = Field(..., description="URL or file path to ingest")
    source_type: SourceType = Field(..., description="Type of source (pdf or html)")


class IngestResponse(ImmutableModel):
    document_id: str
    chunks_created: int = Field(..., ge=0)


class AskRequest(ImmutableModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    filters: dict[str, Any] = Field(default_factory=dict, description="Optional metadata filters")


class AskResponse(ImmutableModel):
    query: str
    answer: str
    citations: list[Citation]
    iterations_taken: int = Field(..., ge=1, description="Number of retrieval loops executed")


class AgentEvent(ImmutableModel):
    """Streaming progress or terminal event emitted by ``AgenticLoopService.ask_stream``."""

    type: AgentEventType = Field(
        ...,
        description="Lifecycle step name (rewrite, retrieve, grade, generate, answer, or error)",
    )
    iteration: int = Field(
        ...,
        ge=0,
        description="1-based retrieve/grade pass number (0 if an error occurs before pass 1)",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Event-specific data: query text for rewrite/retrieve, is_context_sufficient and "
            "reasoning for grade, AskResponse fields for answer, or safe detail for error"
        ),
    )
