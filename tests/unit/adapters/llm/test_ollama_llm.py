import json
from collections.abc import Callable

import httpx
import pytest

from agentic_rag_copilot.adapters.llm import LLMError, OllamaLLM
from agentic_rag_copilot.domain.models import GradeResult
from agentic_rag_copilot.domain.protocols import LLMProtocol

Handler = Callable[[httpx.Request], httpx.Response]
BASE_URL = "http://ollama.test"


def llm_with(handler: Handler) -> OllamaLLM:
    client = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    return OllamaLLM(client, model="test-model")


def respond(text: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": text})

    return handler


def test_implements_protocol() -> None:
    # When
    llm = llm_with(respond("ignored"))

    # Then
    assert isinstance(llm, LLMProtocol)


async def test_generate_text_posts_model_and_prompt() -> None:
    # Given
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"response": "pong"})

    # When
    text = await llm_with(handler).generate_text("ping")

    # Then
    assert text == "pong"
    assert str(requests[0].url) == f"{BASE_URL}/api/generate"
    body = json.loads(requests[0].content)
    assert body == {"model": "test-model", "prompt": "ping", "stream": False}


async def test_generate_text_includes_system_prompt_when_given() -> None:
    # Given
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"response": "ok"})

    # When
    await llm_with(handler).generate_text("ping", system_prompt="be terse")

    # Then
    assert json.loads(requests[0].content)["system"] == "be terse"


async def test_generate_structured_requests_schema_and_validates_response() -> None:
    # Given
    requests: list[httpx.Request] = []
    grade = GradeResult(is_relevant=True, reasoning="Contains the answer.")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"response": grade.model_dump_json()})

    # When
    result = await llm_with(handler).generate_structured("grade this", GradeResult)

    # Then
    assert result == grade
    body = json.loads(requests[0].content)
    assert body["format"] == GradeResult.model_json_schema()


async def test_connection_failure_raises_llm_error() -> None:
    # Given
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    # When / Then
    with pytest.raises(LLMError, match="connection refused"):
        await llm_with(handler).generate_text("ping")


async def test_http_error_reports_status_and_body() -> None:
    # Given
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text='{"error":"model not found"}')

    # When / Then
    with pytest.raises(LLMError, match=r"HTTP 404.*model not found"):
        await llm_with(handler).generate_text("ping")


@pytest.mark.parametrize(
    "body",
    ["not json", '{"unexpected": []}'],
)
async def test_malformed_response_raises_llm_error(body: str) -> None:
    # Given
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    # When / Then
    with pytest.raises(LLMError, match="malformed"):
        await llm_with(handler).generate_text("ping")


async def test_generate_structured_raises_llm_error_on_schema_mismatch() -> None:
    # Given
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "{}"})

    # When / Then
    with pytest.raises(LLMError, match="did not match schema GradeResult"):
        await llm_with(handler).generate_structured("grade this", GradeResult)
