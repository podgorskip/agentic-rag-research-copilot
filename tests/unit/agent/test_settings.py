from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag_copilot.agent import AgentSettings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep a developer's own .env and RAG_* variables from influencing the tests."""
    monkeypatch.chdir(tmp_path)
    for name in AgentSettings.model_fields:
        monkeypatch.delenv(f"RAG_{name.upper()}", raising=False)


def test_defaults() -> None:
    # When
    settings = AgentSettings()

    # Then
    assert settings.llm_provider == "ollama"
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.llm_model == "llama3.2"
    assert settings.llm_timeout_seconds == 120.0
    assert settings.max_loop_count == 3


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setenv("RAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("RAG_LLM_MODEL", "qwen2.5")
    monkeypatch.setenv("RAG_MAX_LOOP_COUNT", "5")

    # When
    settings = AgentSettings()

    # Then
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "qwen2.5"
    assert settings.max_loop_count == 5


def test_rejects_non_positive_max_loop_count() -> None:
    # When / Then
    with pytest.raises(ValidationError):
        AgentSettings(max_loop_count=0)
