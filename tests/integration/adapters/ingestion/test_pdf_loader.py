from pathlib import Path

import pymupdf
import pytest

from agentic_rag_copilot.adapters.ingestion.pdf_loader import PyMuPDFLoader


@pytest.fixture
def dummy_pdf_path(tmp_path: Path) -> str:
    """Programmatically generates a real PDF file for testing."""
    file_path = tmp_path / "test_document.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is a test PDF document for the Agentic RAG Copilot.")
    doc.save(str(file_path))
    doc.close()

    return str(file_path)


@pytest.fixture
def empty_pdf_path(tmp_path: Path) -> str:
    """Generates a valid PDF with zero text (e.g., scanned image only)."""
    file_path = tmp_path / "empty_document.pdf"

    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(file_path))
    doc.close()

    return str(file_path)


async def test_integration_load_valid_pdf_from_disk(dummy_pdf_path: str) -> None:
    loader = PyMuPDFLoader()
    document = await loader.load(dummy_pdf_path)

    assert "Agentic RAG Copilot" in document.raw_content
    assert document.metadata["page_count"] == 1
    assert document.source_uri == dummy_pdf_path


async def test_integration_load_raises_value_error_for_empty_pdf(empty_pdf_path: str) -> None:
    loader = PyMuPDFLoader()

    with pytest.raises(ValueError, match="No extractable text found"):
        await loader.load(empty_pdf_path)


async def test_integration_load_raises_value_error_for_corrupt_file(tmp_path: Path) -> None:
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_text("This is not a real PDF byte structure.")

    loader = PyMuPDFLoader()

    with pytest.raises(ValueError, match="Failed to parse PDF"):
        await loader.load(str(corrupt_file))
