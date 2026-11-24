"""FastAPI application factory. Builds every backing resource once, in the lifespan."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_rag_copilot.domain.exceptions import EmbeddingError, LLMError
from agentic_rag_copilot.ingestion import IngestionSettings
from agentic_rag_copilot.ingestion.cli import build_pipeline
from agentic_rag_copilot.logging import get_logger

from ._dependencies import IngestionSettingsDep, ProbeClientDep
from ._ingest_routes import router as ingest_router
from ._settings import APISettings

_logger = get_logger(__name__)

_PROBE_TIMEOUT_SECONDS: Final[float] = 2.0
_UNAVAILABLE_DETAIL: Final[str] = "A backing service is unavailable; try again later"


def create_app(
    settings: APISettings | None = None,
    ingestion_settings: IngestionSettings | None = None,
) -> FastAPI:
    """Build the HTTP API.

    Args:
        settings: API configuration; read from the environment when omitted.
        ingestion_settings: Qdrant and Ollama configuration; read from the environment when omitted.

    Returns:
        A FastAPI application whose resources are created when the server starts.
    """
    api_settings = settings if settings is not None else APISettings()
    ingest_settings = ingestion_settings if ingestion_settings is not None else IngestionSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with (
            httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as probe_client,
            build_pipeline(ingest_settings) as pipeline,
        ):
            app.state.api_settings = api_settings
            app.state.ingestion_settings = ingest_settings
            app.state.probe_client = probe_client
            app.state.ingestion_pipeline = pipeline
            yield

    app = FastAPI(title="Agentic RAG Copilot", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(EmbeddingError, _unavailable_handler)
    app.add_exception_handler(LLMError, _unavailable_handler)
    app.add_api_route("/health", _health, methods=["GET"])
    app.include_router(ingest_router)
    return app


async def _unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map infrastructure failures to 503 without leaking the underlying error text."""
    _logger.error(f"{request.method} {request.url.path} failed: {exc}")
    return JSONResponse(status_code=503, content={"detail": _UNAVAILABLE_DETAIL})


async def _health(probe: ProbeClientDep, ingest: IngestionSettingsDep) -> dict[str, object]:
    """Report the API status and whether Qdrant and Ollama currently answer."""
    return {
        "status": "ok",
        "qdrant": await _is_reachable(probe, f"{ingest.qdrant_url}/readyz"),
        "ollama": await _is_reachable(probe, f"{ingest.ollama_url}/api/tags"),
    }


async def _is_reachable(client: httpx.AsyncClient, url: str) -> bool:
    try:
        response = await client.get(url)
    except httpx.HTTPError:
        return False
    return response.is_success
