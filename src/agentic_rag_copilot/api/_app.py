"""FastAPI application factory. Builds every backing resource once, in the lifespan."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langfuse import Langfuse
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.adapters.embedding import OllamaEmbedder
from agentic_rag_copilot.adapters.ingestion import (
    LangchainTextChunker,
    PyMuPDFLoader,
    TrafilaturaHTMLLoader,
)
from agentic_rag_copilot.adapters.llm import OllamaLLM
from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker
from agentic_rag_copilot.adapters.vector_store import QdrantVectorStore
from agentic_rag_copilot.agent import (
    AgenticLoopService,
    AgentSettings,
    AnswerSynthesizer,
    QueryRewriter,
    RelevanceGrader,
)
from agentic_rag_copilot.domain.exceptions import EmbeddingError, LLMError
from agentic_rag_copilot.domain.models import SourceType
from agentic_rag_copilot.domain.protocols import (
    EmbedderProtocol,
    LLMProtocol,
    RerankerProtocol,
)
from agentic_rag_copilot.ingestion import IngestionPipeline, IngestionSettings
from agentic_rag_copilot.logging import get_logger
from agentic_rag_copilot.services import RetrievalService

from ._ask_routes import router as ask_router
from ._dependencies import IngestionSettingsDep, ProbeClientDep
from ._ingest_routes import router as ingest_router
from ._settings import APISettings

_logger = get_logger(__name__)

_PROBE_TIMEOUT_SECONDS: Final[float] = 2.0
_UNAVAILABLE_DETAIL: Final[str] = "A backing service is unavailable; try again later"


def create_app(
    settings: APISettings | None = None,
    ingestion_settings: IngestionSettings | None = None,
    agent_settings: AgentSettings | None = None,
    *,
    qdrant_client: AsyncQdrantClient | None = None,
    embedder: EmbedderProtocol | None = None,
    llm: LLMProtocol | None = None,
    reranker: RerankerProtocol | None = None,
) -> FastAPI:
    """Build the HTTP API.

    Args:
        settings: API configuration; read from the environment when omitted.
        ingestion_settings: Qdrant and Ollama configuration; read from the environment when omitted.
        agent_settings: Agent LLM and loop configuration; read from the environment when omitted.
        qdrant_client: Optional pre-configured Qdrant client (e.g. in-memory for integration tests).
            When provided, ownership stays with the caller and the lifespan will not close it.
        embedder: Optional ``EmbedderProtocol`` override for integration tests.
        llm: Optional ``LLMProtocol`` override for integration tests.
        reranker: Optional ``RerankerProtocol`` override for integration tests.

    Returns:
        A FastAPI application whose resources are created when the server starts.
    """
    api_settings = settings if settings is not None else APISettings()
    ingest_settings = ingestion_settings if ingestion_settings is not None else IngestionSettings()
    resolved_agent_settings = agent_settings if agent_settings is not None else AgentSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with (
            httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as probe_client,
            _build_services(
                ingest_settings,
                resolved_agent_settings,
                qdrant_client=qdrant_client,
                embedder=embedder,
                llm=llm,
                reranker=reranker,
            ) as (pipeline, loop_service),
        ):
            app.state.api_settings = api_settings
            app.state.ingestion_settings = ingest_settings
            app.state.probe_client = probe_client
            app.state.ingestion_pipeline = pipeline
            app.state.agentic_loop_service = loop_service
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
    app.include_router(ask_router)
    return app


@asynccontextmanager
async def _build_services(
    ingest_settings: IngestionSettings,
    agent_settings: AgentSettings,
    *,
    qdrant_client: AsyncQdrantClient | None = None,
    embedder: EmbedderProtocol | None = None,
    llm: LLMProtocol | None = None,
    reranker: RerankerProtocol | None = None,
) -> AsyncGenerator[tuple[IngestionPipeline, AgenticLoopService]]:
    """Wire shared Qdrant/Ollama clients into ``IngestionPipeline`` and ``AgenticLoopService``.

    Args:
        ingest_settings: Qdrant, chunking, and embedding connection settings.
        agent_settings: Ollama LLM and loop iteration settings.
        qdrant_client: Optional caller-owned Qdrant client override.
        embedder: Optional embedder override.
        llm: Optional LLM override.
        reranker: Optional reranker override.

    Yields:
        Tuple of ``(IngestionPipeline, AgenticLoopService)`` sharing a single Qdrant client
        and embedder, closed cleanly on shutdown.
    """
    langfuse = _create_langfuse_client()
    chunker = LangchainTextChunker(ingest_settings.chunk_size, ingest_settings.chunk_overlap)
    owns_qdrant = qdrant_client is None
    qdrant = (
        qdrant_client
        if qdrant_client is not None
        else AsyncQdrantClient(url=ingest_settings.qdrant_url, check_compatibility=False)
    )
    try:
        async with (
            httpx.AsyncClient(
                base_url=ingest_settings.ollama_url,
                timeout=ingest_settings.embedding_timeout_seconds,
            ) as embed_client,
            httpx.AsyncClient(
                base_url=agent_settings.ollama_url,
                timeout=agent_settings.llm_timeout_seconds,
            ) as llm_client,
        ):
            resolved_embedder: EmbedderProtocol = (
                embedder
                if embedder is not None
                else OllamaEmbedder(
                    embed_client,
                    model=ingest_settings.embedding_model,
                    batch_size=ingest_settings.embedding_batch_size,
                )
            )
            vector_store = QdrantVectorStore(
                qdrant,
                embedder=resolved_embedder,
                collection_name=ingest_settings.qdrant_collection,
            )
            pipeline = IngestionPipeline(
                loaders={
                    SourceType.PDF: PyMuPDFLoader(),
                    SourceType.HTML: TrafilaturaHTMLLoader(),
                },
                chunker=chunker,
                embedder=resolved_embedder,
                vector_store=vector_store,
            )
            resolved_reranker: RerankerProtocol = (
                reranker if reranker is not None else FastEmbedReranker()
            )
            retrieval_service = RetrievalService(
                embedder=resolved_embedder,
                vector_store=vector_store,
                reranker=resolved_reranker,
                langfuse_client=langfuse,
            )
            resolved_llm: LLMProtocol = (
                llm if llm is not None else OllamaLLM(llm_client, model=agent_settings.llm_model)
            )
            loop_service = AgenticLoopService(
                retrieval_service=retrieval_service,
                grader=RelevanceGrader(resolved_llm),
                rewriter=QueryRewriter(resolved_llm),
                synthesizer=AnswerSynthesizer(resolved_llm),
                settings=agent_settings,
                langfuse_client=langfuse,
            )
            yield pipeline, loop_service
    finally:
        if owns_qdrant:
            await qdrant.close()


def _create_langfuse_client() -> Langfuse:
    """Create a Langfuse client, disabling tracing when credentials are not configured.

    Returns:
        Configured ``Langfuse`` instance.
    """
    has_keys = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    return Langfuse(tracing_enabled=has_keys)


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
