"""Domain-level exceptions for the agentic RAG copilot."""


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend is unreachable or returns an unusable response."""


class LLMError(RuntimeError):
    """Raised when the LLM backend is unreachable or returns an unusable response."""
