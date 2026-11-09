"""Domain-level exceptions for the agentic RAG copilot."""


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend is unreachable or returns an unusable response."""
