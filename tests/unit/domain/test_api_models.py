import pytest
from pydantic import ValidationError

from agentic_rag_copilot.domain.models.api import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
)
from agentic_rag_copilot.domain.models.entities import Citation, SourceType


def test_ingest_models() -> None:
    req = IngestRequest(
        source_uri="https://arxiv.org/pdf/2401.00001.pdf",
        source_type=SourceType.PDF,
    )
    assert req.source_type == SourceType.PDF

    res = IngestResponse(document_id="doc-123", chunks_created=42)
    assert res.chunks_created == 42

    with pytest.raises(ValidationError):
        IngestResponse(document_id="doc-123", chunks_created=-5)


def test_ask_request_validation() -> None:
    req = AskRequest(query="Explain Ragas faithfulness.")
    assert req.top_k == 5  # Default

    # top_k out of bounds (le=20)
    with pytest.raises(ValidationError):
        AskRequest(query="Explain Ragas faithfulness.", top_k=50)

    # Empty query
    with pytest.raises(ValidationError):
        AskRequest(query="")


def test_ask_response() -> None:
    citation = Citation(
        chunk_id="c1",
        document_id="d1",
        source_uri="https://docs.ragas.io",
        quote="Faithfulness measures factual consistency.",
    )
    res = AskResponse(
        query="What is faithfulness?",
        answer="It measures factual consistency [1].",
        citations=[citation],
        iterations_taken=2,
    )
    assert res.iterations_taken == 2
    assert len(res.citations) == 1
