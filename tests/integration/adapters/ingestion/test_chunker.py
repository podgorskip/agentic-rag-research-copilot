import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast

import pytest

from agentic_rag_copilot.adapters.ingestion import LangchainTextChunker, TrafilaturaHTMLLoader


class PipelineMockHandler(BaseHTTPRequestHandler):
    """HTTP request handler that serves a large HTML document for testing."""

    def do_GET(self) -> None:
        """Handle GET requests by returning a large HTML document.

        Returns:
            200 OK response with HTML content containing repeated paragraph elements.
        """
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")

        long_text = "<p>Word </p>" * 500
        html = f"<html><body><main>{long_text}</main></body></html>"

        body = html.encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress HTTP request logging.

        Args:
            format: Log message format string (unused).
            *args: Format arguments (unused).
        """
        pass


@pytest.fixture(scope="module")
def pipeline_server() -> Iterator[str]:
    """Create a temporary HTTP server serving test HTML content.

    Returns:
        Server URL (e.g., "http://127.0.0.1:12345").

    Yields:
        URL for the duration of the test module, then shuts down the server.
    """
    server = HTTPServer(("127.0.0.1", 0), cast("Any", PipelineMockHandler))
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()
    thread.join()


async def test_end_to_end_ingestion_pipeline(pipeline_server: str) -> None:
    """Test the complete ingestion pipeline from HTML loading to document chunking.

    Args:
        pipeline_server: URL of test HTTP server providing HTML content.

    Returns:
        Multiple chunks from the loaded and processed document with proper metadata.
    """
    loader = TrafilaturaHTMLLoader()
    chunker = LangchainTextChunker(chunk_size=200, chunk_overlap=50)

    document = await loader.load(pipeline_server)

    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    assert chunks[0].metadata["parser"] == "trafilatura"
    assert chunks[0].document_id == document.id
    assert chunks[0].metadata["chunk_index"] == 0
