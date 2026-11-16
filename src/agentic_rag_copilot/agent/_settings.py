from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Agent loop configuration, read from ``RAG_``-prefixed variables or a local ``.env``."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file="config/.env", extra="ignore")

    llm_provider: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    max_loop_count: int = Field(default=3, ge=1, description="Caps multi-hop self-correction")
