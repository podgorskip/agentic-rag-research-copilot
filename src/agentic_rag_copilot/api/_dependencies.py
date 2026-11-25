from typing import Annotated

import httpx
from fastapi import Depends, Request

from agentic_rag_copilot.agent import AgenticLoopService
from agentic_rag_copilot.ingestion import IngestionPipeline, IngestionSettings

from ._settings import APISettings


def get_api_settings(request: Request) -> APISettings:
    """Return the ``APISettings`` instance stored on application state.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Configured ``APISettings``.
    """
    return request.app.state.api_settings


def get_ingestion_settings(request: Request) -> IngestionSettings:
    """Return the ``IngestionSettings`` instance stored on application state.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Configured ``IngestionSettings``.
    """
    return request.app.state.ingestion_settings


def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    """Return the ``IngestionPipeline`` instance stored on application state.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Configured ``IngestionPipeline``.
    """
    return request.app.state.ingestion_pipeline


def get_agentic_loop_service(request: Request) -> AgenticLoopService:
    """Return the ``AgenticLoopService`` instance stored on application state.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Configured ``AgenticLoopService``.
    """
    return request.app.state.agentic_loop_service


def get_probe_client(request: Request) -> httpx.AsyncClient:
    """Return the upstream health-probe ``httpx.AsyncClient`` from application state.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Configured ``httpx.AsyncClient``.
    """
    return request.app.state.probe_client


ApiSettingsDep = Annotated[APISettings, Depends(get_api_settings)]
IngestionSettingsDep = Annotated[IngestionSettings, Depends(get_ingestion_settings)]
IngestionPipelineDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]
AgenticLoopServiceDep = Annotated[AgenticLoopService, Depends(get_agentic_loop_service)]
ProbeClientDep = Annotated[httpx.AsyncClient, Depends(get_probe_client)]
