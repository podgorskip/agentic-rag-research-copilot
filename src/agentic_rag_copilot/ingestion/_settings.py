from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """Ingestion configuration, read from ``RAG_``-prefixed variables or a local ``.env``."""

    # .env also holds docker-compose variables that have no matching field here.
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_timeout_seconds: float = Field(default=120.0, gt=0)
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)
