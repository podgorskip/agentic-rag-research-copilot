import pytest
from pydantic import ValidationError

from agentic_rag_copilot.domain.models import Chunk, Citation, Document, Query, SourceType


def test_document_creation_and_immutability() -> None:
    doc = Document(
        id="doc-1",
        source_type=SourceType.PDF,
        source_uri="https://arxiv.org/pdf/1234.5678.pdf",
        title="Agentic RAG Paper",
        raw_content="Full text of the paper.",
    )
    assert doc.id == "doc-1"
    assert doc.source_type == SourceType.PDF
    assert doc.metadata == {}

    with pytest.raises(ValidationError):
        doc.title = "Mutated Title"


def test_document_strict_validation_fails_on_wrong_types() -> None:
    with pytest.raises(ValidationError):
        Document(
            id="doc-1",
            source_type=SourceType.HTML,
            source_uri="https://docs.example.com",
            raw_content=12345,
        )

    with pytest.raises(ValidationError):
        Document(
            id="doc-1",
            source_type=SourceType.HTML,
            source_uri="https://docs.example.com",
            raw_content="",
        )


def test_chunk_creation() -> None:
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        text="A specific section of the paper.",
        embedding=[0.1, 0.2, 0.3],
        score=0.95,
    )
    assert chunk.document_id == "doc-1"
    assert chunk.score == 0.95
    assert chunk.embedding == [0.1, 0.2, 0.3]


def test_query_creation() -> None:
    query = Query(text="How does hybrid search work?", filters={"year": 2026})
    assert query.text == "How does hybrid search work?"
    assert query.filters["year"] == 2026
    assert query.embedding is None


def test_citation_creation() -> None:
    citation = Citation(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_uri="https://arxiv.org/pdf/1234.5678.pdf",
        quote="Hybrid search combines BM25 and dense vectors.",
    )
    assert citation.chunk_id == "chunk-1"
    assert "BM25" in citation.quote
