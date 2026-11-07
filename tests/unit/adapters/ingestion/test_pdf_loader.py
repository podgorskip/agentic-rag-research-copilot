from unittest.mock import MagicMock, patch

import pytest

from agentic_rag_copilot.adapters.ingestion.pdf_loader import PyMuPDFLoader
from agentic_rag_copilot.domain.models import SourceType
from agentic_rag_copilot.domain.protocols.ingestion import DocumentLoaderProtocol


def test_pdf_loader_implements_protocol() -> None:
    loader = PyMuPDFLoader()
    assert isinstance(loader, DocumentLoaderProtocol)


@patch("agentic_rag_copilot.adapters.ingestion.pdf_loader.Path.exists")
async def test_load_raises_file_not_found(mock_exists: MagicMock) -> None:
    mock_exists.return_value = False
    loader = PyMuPDFLoader()

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        await loader.load("dummy_missing.pdf")


@patch("agentic_rag_copilot.adapters.ingestion.pdf_loader.pymupdf.open")
@patch("agentic_rag_copilot.adapters.ingestion.pdf_loader.Path.exists")
async def test_load_valid_pdf_extracts_content(
    mock_exists: MagicMock, mock_pymupdf_open: MagicMock
) -> None:
    mock_exists.return_value = True

    # Configure the mock PyMuPDF document context manager
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Mocked PDF content."

    # Simulate iterating over pages in the document
    mock_doc.__iter__.return_value = [mock_page]
    mock_doc.page_count = 1
    mock_doc.metadata = {"author": "John Doe", "title": "Test Paper"}

    mock_pymupdf_open.return_value.__enter__.return_value = mock_doc

    loader = PyMuPDFLoader()
    document = await loader.load("dummy.pdf")

    assert document.source_type == SourceType.PDF
    assert document.raw_content == "Mocked PDF content."
    assert document.metadata["page_count"] == 1
    assert document.metadata["author"] == "John Doe"
    assert len(document.id) == 64  # Deterministic SHA-256 hash


@patch("agentic_rag_copilot.adapters.ingestion.pdf_loader.pymupdf.open")
@patch("agentic_rag_copilot.adapters.ingestion.pdf_loader.Path.exists")
async def test_load_raises_value_error_on_library_exception(
    mock_exists: MagicMock, mock_pymupdf_open: MagicMock
) -> None:
    mock_exists.return_value = True
    # Simulate PyMuPDF throwing an internal error (e.g., FileDataError)
    mock_pymupdf_open.side_effect = Exception("Internal PyMuPDF Error")

    loader = PyMuPDFLoader()
    with pytest.raises(ValueError, match=r"Failed to parse PDF dummy\.pdf: Internal PyMuPDF Error"):
        await loader.load("dummy.pdf")
