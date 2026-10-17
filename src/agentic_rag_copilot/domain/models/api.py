from typing import Any

from pydantic import Field

from .base import ImmutableModel
from .entities import Citation, SourceType


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
