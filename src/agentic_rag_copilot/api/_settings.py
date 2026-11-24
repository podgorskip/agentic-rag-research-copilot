from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class APISettings(BaseSettings):
    """HTTP API configuration, read from ``RAG_API_``-prefixed variables or a local ``.env``."""

    model_config = SettingsConfigDict(env_prefix="RAG_API_", env_file="config/.env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501"],
        description="Origins allowed to call the API (the Streamlit UI by default)",
    )
    max_upload_bytes: int = Field(default=_DEFAULT_MAX_UPLOAD_BYTES, ge=1)
