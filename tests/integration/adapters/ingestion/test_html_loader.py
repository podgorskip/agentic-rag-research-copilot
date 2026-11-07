import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast

import pytest

from agentic_rag_copilot.adapters.ingestion import TrafilaturaHTMLLoader
from agentic_rag_copilot.domain.models import SourceType


class MockDocumentationHandler(BaseHTTPRequestHandler):
    """Serves mock HTML payloads based on the requested path."""

    def do_GET(self) -> None:
        if self.path == "/valid":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            html = """
            <html>
                <head><title>Mock Docs</title></head>
                <body>
                    <header>
                        <nav><ul><li><a href="/home">Do not extract me</a></li></ul></nav>
                    </header>
                    <main>
                        <article>
                            <h1>Main Title</h1>
                            <p>This is the core documentation text that should be extracted.
                            It needs to have enough text density so the scraper recognizes
                            it as the primary content rather than a random snippet.</p>
                            <p>Here is a second paragraph to further increase
                            the text-to-tag ratio.</p>
                        </article>
                    </main>
                    <footer>Copyright 2026</footer>
                </body>
            </html>
            """
            body = html.encode("utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/empty":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            body = b"<html><body></body></html>"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP server logging to keep test output clean."""
        pass


@pytest.fixture(scope="module")
def local_http_server() -> Iterator[str]:
    """Spins up a local HTTP server for the duration of the module tests."""
    # Use 127.0.0.1 to avoid Windows localhost IPv4/IPv6 resolution issues
    # Port 0 asks the OS to assign a random free port automatically
    server = HTTPServer(("127.0.0.1", 0), cast("Any", MockDocumentationHandler))
    port = server.server_port

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()
    thread.join()


async def test_integration_load_valid_html_from_network(local_http_server: str) -> None:
    loader = TrafilaturaHTMLLoader()
    document = await loader.load(f"{local_http_server}/valid")

    assert document.source_type == SourceType.HTML
    assert "This is the core documentation text" in document.raw_content
    # Ensure boilerplate was stripped
    assert "Do not extract me" not in document.raw_content
    assert "Copyright" not in document.raw_content


async def test_integration_load_raises_value_error_for_404(local_http_server: str) -> None:
    loader = TrafilaturaHTMLLoader()

    with pytest.raises(ValueError, match="Failed to fetch HTML from"):
        await loader.load(f"{local_http_server}/missing")


async def test_integration_load_raises_value_error_for_empty_page(local_http_server: str) -> None:
    loader = TrafilaturaHTMLLoader()

    with pytest.raises(ValueError, match="Failed to extract meaningful text"):
        await loader.load(f"{local_http_server}/empty")
