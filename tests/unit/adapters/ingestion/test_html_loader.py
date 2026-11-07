from unittest.mock import MagicMock, patch

import pytest

from agentic_rag_copilot.adapters.ingestion import TrafilaturaHTMLLoader
from agentic_rag_copilot.domain.models import SourceType
from agentic_rag_copilot.domain.protocols import DocumentLoaderProtocol


def test_html_loader_implements_protocol() -> None:
    loader = TrafilaturaHTMLLoader()
    assert isinstance(loader, DocumentLoaderProtocol)


@patch("agentic_rag_copilot.adapters.ingestion._html_loader.trafilatura.extract")
@patch("agentic_rag_copilot.adapters.ingestion._html_loader.urllib.request.urlopen")
async def test_load_valid_html_extracts_content(
    mock_urlopen: MagicMock, mock_extract: MagicMock
) -> None:
    # Setup mock HTTP response
    mock_response = MagicMock()
    mock_response.read.return_value = b"<html><body><p>Mocked raw HTML</p></body></html>"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    mock_extract.return_value = "Cleaned documentation text."

    loader = TrafilaturaHTMLLoader()
    document = await loader.load("https://docs.python.org/3/library/asyncio.html")

    assert document.source_type == SourceType.HTML
    assert document.raw_content == "Cleaned documentation text."
    assert document.metadata["parser"] == "trafilatura"
    assert len(document.id) == 64


@patch("agentic_rag_copilot.adapters.ingestion._html_loader.urllib.request.urlopen")
async def test_load_raises_value_error_on_fetch_failure(mock_urlopen: MagicMock) -> None:
    mock_urlopen.side_effect = Exception("Connection Refused")

    loader = TrafilaturaHTMLLoader()
    with pytest.raises(ValueError, match="Failed to fetch HTML from"):
        await loader.load("https://invalid.example.com")


@patch("agentic_rag_copilot.adapters.ingestion._html_loader.trafilatura.extract")
@patch("agentic_rag_copilot.adapters.ingestion._html_loader.urllib.request.urlopen")
async def test_load_raises_value_error_on_extraction_failure(
    mock_urlopen: MagicMock, mock_extract: MagicMock
) -> None:
    mock_response = MagicMock()
    mock_response.read.return_value = b"<html><body></body></html>"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    mock_extract.return_value = None

    loader = TrafilaturaHTMLLoader()
    with pytest.raises(ValueError, match="Failed to extract meaningful text from"):
        await loader.load("https://empty.example.com")


@patch("agentic_rag_copilot.adapters.ingestion._html_loader.trafilatura.extract")
@patch("agentic_rag_copilot.adapters.ingestion._html_loader.urllib.request.urlopen")
async def test_load_raises_value_error_on_parser_exception(
    mock_urlopen: MagicMock, mock_extract: MagicMock
) -> None:
    mock_response = MagicMock()
    mock_response.read.return_value = b"<html><body>broken</body></html>"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    mock_extract.side_effect = Exception("Simulated LXML crash")

    loader = TrafilaturaHTMLLoader()

    expected_error = (
        r"Failed to extract meaningful text from: https://broken\.example\.com\. "
        r"Parser error: Simulated LXML crash"
    )

    with pytest.raises(ValueError, match=expected_error):
        await loader.load("https://broken.example.com")
