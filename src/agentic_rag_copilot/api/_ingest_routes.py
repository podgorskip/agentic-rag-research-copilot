"""``POST /ingest``: ingest a source given as a JSON URI or as a multipart PDF/HTML upload."""

import hashlib
import tempfile
from pathlib import Path
from typing import Final

import anyio
from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile

from agentic_rag_copilot.domain.models import IngestRequest, IngestResponse
from agentic_rag_copilot.ingestion import IngestionError, IngestionPipeline, detect_source_type
from agentic_rag_copilot.logging import get_logger

from ._dependencies import ApiSettingsDep, IngestionPipelineDep
from ._settings import APISettings

_logger = get_logger(__name__)

router = APIRouter()

_UPLOAD_DIR_NAME: Final[str] = "rag-copilot-uploads"
_READ_CHUNK_BYTES: Final[int] = 1024 * 1024
# Allowance for multipart framing (boundaries and headers) on top of the file itself.
_MULTIPART_OVERHEAD_BYTES: Final[int] = 64 * 1024


@router.post("/ingest", status_code=201, response_model=IngestResponse)
async def ingest(
    request: Request, pipeline: IngestionPipelineDep, settings: ApiSettingsDep
) -> IngestResponse:
    """Ingest one source, from a JSON ``IngestRequest`` or a multipart ``file`` upload.

    Args:
        request: The raw request, whose content type selects the body format.
        pipeline: Ingestion pipeline that loads, chunks, embeds and stores the source.
        settings: API limits, including the maximum upload size.

    Returns:
        The stored document ID and the number of chunks created.

    Raises:
        HTTPException: 413 for an oversized upload, 422 for an invalid request or a source
            that could not be ingested.
    """
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        return await _ingest_upload(request, pipeline, settings)
    return await _ingest_source(await _parse_json_body(request), pipeline)


async def _parse_json_body(request: Request) -> IngestRequest:
    try:
        return IngestRequest.model_validate_json(await request.body())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid ingest request: {exc}") from exc


async def _ingest_upload(
    request: Request, pipeline: IngestionPipeline, settings: APISettings
) -> IngestResponse:
    declared_size = int(request.headers.get("content-length", "0"))
    if declared_size > settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES:
        raise _too_large(settings.max_upload_bytes)

    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(status_code=422, detail="Multipart request must include a 'file' field")

    try:
        source_type = detect_source_type(upload.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    stored = await _store_upload(upload, settings.max_upload_bytes)
    try:
        return await _ingest_source(
            IngestRequest(source_uri=str(stored), source_type=source_type), pipeline
        )
    finally:
        await anyio.Path(stored).unlink(missing_ok=True)


async def _store_upload(upload: UploadFile, max_bytes: int) -> Path:
    """Stream an upload to disk under a content-addressed name, so re-uploads keep their ID.

    The loaders derive the document ID from the source path, so naming the file by its SHA-256
    makes uploading the same bytes twice address the same document.

    Args:
        upload: The uploaded file.
        max_bytes: Largest accepted file size in bytes.

    Returns:
        Path of the stored file; the caller is responsible for deleting it.

    Raises:
        HTTPException: 413 when the upload exceeds ``max_bytes``.
    """
    suffix = Path(upload.filename or "").suffix.lower()
    digest = hashlib.sha256()
    total = 0

    upload_dir = anyio.Path(tempfile.gettempdir()) / _UPLOAD_DIR_NAME
    await upload_dir.mkdir(parents=True, exist_ok=True)
    async with anyio.NamedTemporaryFile(dir=str(upload_dir), suffix=suffix, delete=False) as staged:
        staged_path = anyio.Path(str(staged.name))
        try:
            while chunk := await upload.read(_READ_CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    raise _too_large(max_bytes)
                digest.update(chunk)
                await staged.write(chunk)
        except BaseException:
            await staged_path.unlink(missing_ok=True)
            raise

    stored = upload_dir / f"{digest.hexdigest()}{suffix}"
    await staged_path.replace(stored)
    return Path(stored)


async def _ingest_source(request: IngestRequest, pipeline: IngestionPipeline) -> IngestResponse:
    try:
        return await pipeline.ingest(request)
    except IngestionError as exc:
        _logger.warning(f"Ingestion failed for {request.source_type}: {exc}")
        # The stage is named but the underlying reason and source path are not echoed back.
        raise HTTPException(
            status_code=422,
            detail=f"Ingestion failed during the {exc.stage} stage",
        ) from exc


def _too_large(max_bytes: int) -> HTTPException:
    return HTTPException(status_code=413, detail=f"Upload exceeds the {max_bytes}-byte limit")
