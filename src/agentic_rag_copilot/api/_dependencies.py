from typing import Annotated

import httpx
from fastapi import Depends, Request

from agentic_rag_copilot.ingestion import IngestionPipeline, IngestionSettings

from ._settings import APISettings


def get_api_settings(request: Request) -> APISettings:
    return request.app.state.api_settings


def get_ingestion_settings(request: Request) -> IngestionSettings:
    return request.app.state.ingestion_settings


def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    return request.app.state.ingestion_pipeline


def get_probe_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.probe_client


ApiSettingsDep = Annotated[APISettings, Depends(get_api_settings)]
IngestionSettingsDep = Annotated[IngestionSettings, Depends(get_ingestion_settings)]
IngestionPipelineDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]
ProbeClientDep = Annotated[httpx.AsyncClient, Depends(get_probe_client)]
