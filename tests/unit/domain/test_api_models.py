import pytest
from pydantic import ValidationError

from agentic_rag_copilot.domain.models import (
    AskRequest,
    AskResponse,
    Citation,
    IngestRequest,
    IngestResponse,
    SourceType,
)


def test_ingest_models() -> None:
    # When
    req = IngestRequest(
        source_uri="https://arxiv.org/pdf/2401.00001.pdf",
        source_type=SourceType.PDF,
    )

    # Then
    assert req.source_type == SourceType.PDF

    # When
    res = IngestResponse(document_id="doc-123", chunks_created=42)

    # Then
    assert res.chunks_created == 42

    # When / Then: a negative chunk count is rejected
    with pytest.raises(ValidationError):
        IngestResponse(document_id="doc-123", chunks_created=-5)


def test_ask_request_validation() -> None:
    # When
    req = AskRequest(query="Explain Ragas faithfulness.")

    # Then
    assert req.top_k == 5  # Default

    # When / Then: top_k above the allowed range is rejected
    with pytest.raises(ValidationError):
        AskRequest(query="Explain Ragas faithfulness.", top_k=50)

    # When / Then: an empty query is rejected
    with pytest.raises(ValidationError):
        AskRequest(query="")


def test_ask_response() -> None:
    # Given
    citation = Citation(
        chunk_id="c1",
        document_id="d1",
        source_uri="https://docs.ragas.io",
        quote="Faithfulness measures factual consistency.",
    )

    # When
    res = AskResponse(
        query="What is faithfulness?",
        answer="It measures factual consistency [1].",
        citations=[citation],
        iterations_taken=2,
    )

    # Then
    assert res.iterations_taken == 2
    assert len(res.citations) == 1
