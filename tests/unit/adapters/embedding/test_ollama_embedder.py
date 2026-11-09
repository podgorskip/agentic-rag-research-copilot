import json
from collections.abc import Callable

import httpx
import pytest

from agentic_rag_copilot.adapters.embedding import EmbeddingError, OllamaEmbedder
from agentic_rag_copilot.domain.protocols import EmbedderProtocol

Handler = Callable[[httpx.Request], httpx.Response]
BASE_URL = "http://ollama.test"


def embedder_with(handler: Handler, batch_size: int = 32) -> OllamaEmbedder:
    client = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    return OllamaEmbedder(client, model="test-model", batch_size=batch_size)


def echo_lengths(request: httpx.Request) -> httpx.Response:
    """Answer with one vector per input, each encoding its input so ordering can be checked."""
    inputs = json.loads(request.content)["input"]
    return httpx.Response(200, json={"embeddings": [[float(text)] for text in inputs]})


def test_implements_protocol() -> None:
    assert isinstance(embedder_with(echo_lengths), EmbedderProtocol)


async def test_embed_text_posts_model_and_input() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    vector = await embedder_with(handler).embed_text("hello")

    assert vector == [0.1, 0.2]
    assert str(requests[0].url) == f"{BASE_URL}/api/embed"
    assert json.loads(requests[0].content) == {
        "model": "test-model",
        "input": ["hello"],
        "truncate": False,
    }


async def test_embed_batch_splits_requests_and_preserves_order() -> None:
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        batch_sizes.append(len(json.loads(request.content)["input"]))
        return echo_lengths(request)

    vectors = await embedder_with(handler, batch_size=2).embed_batch(["1", "2", "3", "4", "5"])

    assert batch_sizes == [2, 2, 1]
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]


async def test_embed_batch_of_nothing_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert await embedder_with(handler).embed_batch([]) == []


async def test_connection_failure_raises_embedding_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(EmbeddingError, match="connection refused"):
        await embedder_with(handler).embed_text("hello")


async def test_http_error_reports_status_and_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text='{"error":"model not found"}')

    with pytest.raises(EmbeddingError, match=r"HTTP 404.*model not found"):
        await embedder_with(handler).embed_text("hello")


@pytest.mark.parametrize(
    "body",
    ["not json", '{"unexpected": []}', '{"embeddings": [["a"]]}'],
)
async def test_malformed_response_raises_embedding_error(body: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    with pytest.raises(EmbeddingError, match="malformed"):
        await embedder_with(handler).embed_text("hello")


async def test_wrong_embedding_count_raises_embedding_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1]]})

    with pytest.raises(EmbeddingError, match="1 embeddings for 2 texts"):
        await embedder_with(handler).embed_batch(["a", "b"])


def test_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        embedder_with(echo_lengths, batch_size=0)
