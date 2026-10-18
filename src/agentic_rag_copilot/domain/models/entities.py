from enum import StrEnum
from typing import Any

from pydantic import Field

from .base import ImmutableModel


class SourceType(StrEnum):
    PDF = "pdf"
    HTML = "html"


class Document(ImmutableModel):
    id: str = Field(..., description="Unique hash or UUID for the document")
    source_type: SourceType = Field(..., description="PDF (arXiv) or HTML (Docs)")
    source_uri: str = Field(..., description="URL or local file path")
    title: str | None = Field(default=None, description="Extracted document title")
    raw_content: str = Field(..., min_length=1, description="Full extracted text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Author, date, etc.")


class Chunk(ImmutableModel):
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent Document ID")
    text: str = Field(..., min_length=1, description="Chunk text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Page numbers, headers, etc."
    )
    embedding: list[float] | None = Field(default=None, description="Dense vector")
    score: float | None = Field(default=None, description="Search or reranker relevance score")


class Citation(ImmutableModel):
    chunk_id: str = Field(..., description="ID of the chunk backing the claim")
    document_id: str = Field(..., description="ID of the source document")
    source_uri: str = Field(..., description="URL or path for user reference")
    quote: str = Field(..., description="Exact text span supporting the answer")


class Query(ImmutableModel):
    text: str = Field(
        ..., min_length=1, description="The search query text (original or rewritten)"
    )
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Metadata filters for retrieval"
    )
    embedding: list[float] | None = Field(
        default=None, description="Dense vector representation of the query"
    )
