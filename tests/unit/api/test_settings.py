from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag_copilot.api import APISettings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own .env and RAG_API_* variables from influencing the tests."""
    monkeypatch.chdir(tmp_path)
    for name in APISettings.model_fields:
        monkeypatch.delenv(f"RAG_API_{name.upper()}", raising=False)


def test_defaults() -> None:
    # When
    settings = APISettings()

    # Then
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.cors_origins == ["http://localhost:8501"]
    assert settings.max_upload_bytes == 50 * 1024 * 1024


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setenv("RAG_API_PORT", "9000")
    monkeypatch.setenv("RAG_API_CORS_ORIGINS", '["https://ui.example.com"]')
    monkeypatch.setenv("RAG_API_MAX_UPLOAD_BYTES", "1024")

    # When
    settings = APISettings()

    # Then
    assert settings.port == 9000
    assert settings.cors_origins == ["https://ui.example.com"]
    assert settings.max_upload_bytes == 1024


def test_invalid_port_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setenv("RAG_API_PORT", "70000")

    # When / Then
    with pytest.raises(ValidationError):
        APISettings()
