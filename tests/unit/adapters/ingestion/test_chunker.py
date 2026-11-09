from unittest.mock import MagicMock

import pytest

from agentic_rag_copilot.adapters.ingestion import LangchainTextChunker
from agentic_rag_copilot.domain.models import Document, SourceType
from agentic_rag_copilot.domain.protocols import ChunkerProtocol


@pytest.fixture
def dummy_document() -> Document:
    """Create a dummy document fixture for testing.

    Returns:
        Document with fixed ID, 100 'A' characters content, and test metadata.
    """
    return Document(
        id="doc_123",
        source_type=SourceType.PDF,
        source_uri="dummy.pdf",
        raw_content="A" * 100,
        metadata={"author": "Test Author"},
    )


def test_chunker_implements_protocol() -> None:
    """Verify that LangchainTextChunker implements the ChunkerProtocol interface."""
    chunker = LangchainTextChunker()
    assert isinstance(chunker, ChunkerProtocol)


def test_chunk_document_smaller_than_chunk_size(dummy_document: Document) -> None:
    """Test that a document smaller than chunk size results in a single chunk.

    Args:
        dummy_document: 100-character document fixture.

    Returns:
        A single chunk with preserved content, metadata, and SHA-256 ID.
    """
    chunker = LangchainTextChunker(chunk_size=150, chunk_overlap=20)

    chunks = chunker.chunk(dummy_document)

    assert len(chunks) == 1
    assert chunks[0].text == dummy_document.raw_content
    assert chunks[0].document_id == dummy_document.id
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["author"] == "Test Author"
    assert len(chunks[0].id) == 64


def test_chunk_intelligent_split_logic() -> None:
    """Test that chunker respects paragraph boundaries in splitting.

    Returns:
        Two chunks split at paragraph boundaries, preserving semantic units.
    """
    doc = Document(
        id="doc_text",
        source_type=SourceType.HTML,
        source_uri="dummy.html",
        raw_content="Paragraph one is here.\n\nParagraph two is here.",
        metadata={},
    )

    chunker = LangchainTextChunker(chunk_size=30, chunk_overlap=0)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 2
    assert chunks[0].text == "Paragraph one is here."
    assert chunks[1].text == "Paragraph two is here."


def test_chunk_empty_document() -> None:
    """Test that chunking an empty document returns an empty list.

    Returns:
        Empty list when document content is empty.
    """
    mock_doc = MagicMock(spec=Document)
    mock_doc.raw_content = ""
    mock_doc.id = "empty_doc"
    mock_doc.metadata = {}

    chunker = LangchainTextChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk(mock_doc)

    assert len(chunks) == 0
