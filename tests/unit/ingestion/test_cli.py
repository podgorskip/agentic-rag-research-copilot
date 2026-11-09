"""Drives the CLI through the real loader, chunker and store; only Ollama is faked."""

import asyncio
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pymupdf
import pytest
from qdrant_client import AsyncQdrantClient

from agentic_rag_copilot.ingestion import IngestionSettings, cli

Handler = Callable[[httpx.Request], httpx.Response]


def embed_ok(request: httpx.Request) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    return httpx.Response(200, json={"embeddings": [[0.6, 0.0, 0.8] for _ in inputs]})


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own .env and RAG_* variables from influencing the tests."""
    monkeypatch.chdir(tmp_path)
    for name in IngestionSettings.model_fields:
        monkeypatch.delenv(f"RAG_{name.upper()}", raising=False)


@pytest.fixture
def qdrant(monkeypatch: pytest.MonkeyPatch) -> Iterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(":memory:")
    close = AsyncMock()
    monkeypatch.setattr(client, "close", close)

    def factory(**_: Any) -> AsyncQdrantClient:
        return client

    monkeypatch.setattr(cli, "AsyncQdrantClient", factory)
    yield client


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], None]:
    real_client = httpx.AsyncClient

    def install(handler: Handler) -> None:
        def factory(**kwargs: Any) -> httpx.AsyncClient:
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(cli.httpx, "AsyncClient", factory)

    return install


@pytest.fixture
def pdf_path(tmp_path: Path) -> str:
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "Self-correcting retrieval improves answers.")
        document.save(str(path))
    return str(path)


def stored_count(client: AsyncQdrantClient) -> int:
    async def count() -> int:
        if not await client.collection_exists("chunks"):
            return 0
        return (await client.count("chunks", exact=True)).count

    return asyncio.run(count())


def test_ingests_pdf_end_to_end(
    qdrant: AsyncQdrantClient,
    fake_ollama: Callable[[Handler], None],
    pdf_path: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ollama(embed_ok)

    assert cli.main([pdf_path]) == cli.EXIT_OK

    assert stored_count(qdrant) == 1
    output = capsys.readouterr().out
    assert f"OK {pdf_path}: 1 chunks" in output
    assert "Ingested 1 of 1 sources" in output


def test_unreadable_file_does_not_abort_the_batch(
    qdrant: AsyncQdrantClient,
    fake_ollama: Callable[[Handler], None],
    pdf_path: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ollama(embed_ok)

    assert cli.main(["missing.pdf", pdf_path]) == cli.EXIT_INGESTION_FAILED

    assert stored_count(qdrant) == 1
    captured = capsys.readouterr()
    assert "FAILED missing.pdf: load stage failed" in captured.err
    assert "Ingested 1 of 2 sources" in captured.out


def test_corrupt_pdf_is_reported(
    qdrant: AsyncQdrantClient,
    fake_ollama: Callable[[Handler], None],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ollama(embed_ok)
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"this is not a pdf")

    assert cli.main([str(corrupt)]) == cli.EXIT_INGESTION_FAILED

    assert "load stage failed" in capsys.readouterr().err
    assert stored_count(qdrant) == 0


def test_unsupported_source_is_reported(
    qdrant: AsyncQdrantClient,
    fake_ollama: Callable[[Handler], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ollama(embed_ok)

    assert cli.main(["notes.txt"]) == cli.EXIT_INGESTION_FAILED

    assert "Cannot determine source type" in capsys.readouterr().err


def test_ollama_network_failure_is_reported(
    qdrant: AsyncQdrantClient,
    fake_ollama: Callable[[Handler], None],
    pdf_path: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    fake_ollama(refuse)

    assert cli.main([pdf_path]) == cli.EXIT_INGESTION_FAILED

    assert "embed stage failed" in capsys.readouterr().err
    assert stored_count(qdrant) == 0


def test_qdrant_failure_is_reported(
    qdrant: AsyncQdrantClient,
    fake_ollama: Callable[[Handler], None],
    pdf_path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_ollama(embed_ok)
    monkeypatch.setattr(qdrant, "upsert", AsyncMock(side_effect=ConnectionError("qdrant down")))

    assert cli.main([pdf_path]) == cli.EXIT_INGESTION_FAILED

    assert "upsert stage failed: qdrant down" in capsys.readouterr().err


def test_qdrant_client_is_closed(
    qdrant: AsyncQdrantClient,
    fake_ollama: Callable[[Handler], None],
    pdf_path: str,
) -> None:
    fake_ollama(embed_ok)

    cli.main([pdf_path])

    assert isinstance(qdrant.close, AsyncMock)
    qdrant.close.assert_awaited_once()


@pytest.mark.parametrize(
    ("variable", "value"),
    [("RAG_EMBEDDING_BATCH_SIZE", "0"), ("RAG_CHUNK_OVERLAP", "5000")],
)
def test_invalid_configuration_exits_with_config_code(
    variable: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(variable, value)

    assert cli.main(["paper.pdf"]) == cli.EXIT_INVALID_CONFIG

    assert "Invalid configuration" in capsys.readouterr().err


def test_requires_at_least_one_source(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main([])

    assert caught.value.code == 2


def test_settings_ignore_unrelated_dotenv_entries(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("QDRANT_VERSION=v1\nRAG_EMBEDDING_MODEL=custom\n")

    assert IngestionSettings().embedding_model == "custom"
