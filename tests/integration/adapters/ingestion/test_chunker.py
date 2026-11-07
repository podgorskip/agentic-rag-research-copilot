import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast

import pytest

from agentic_rag_copilot.adapters.ingestion import LangchainTextChunker, TrafilaturaHTMLLoader


class PipelineMockHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")

        # A long article to guarantee multiple chunks are generated
        long_text = "<p>Word </p>" * 500
        html = f"<html><body><main>{long_text}</main></body></html>"

        body = html.encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def pipeline_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), cast("Any", PipelineMockHandler))
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()
    thread.join()


async def test_end_to_end_ingestion_pipeline(pipeline_server: str) -> None:
    # Initialize pipeline adapters
    loader = TrafilaturaHTMLLoader()
    chunker = LangchainTextChunker(chunk_size=200, chunk_overlap=50)

    # Execute Loader (Async)
    document = await loader.load(pipeline_server)

    # Execute Chunker (Sync)
    chunks = chunker.chunk(document)

    # Verify Pipeline Results
    assert len(chunks) > 1
    assert chunks[0].metadata["parser"] == "trafilatura"
    assert chunks[0].document_id == document.id
    assert chunks[0].metadata["chunk_index"] == 0
