import httpx
from pydantic import BaseModel, ValidationError

from agentic_rag_copilot.domain.exceptions import EmbeddingError
from agentic_rag_copilot.domain.protocols import EmbedderProtocol

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_BATCH_SIZE = 32


class _EmbedResponse(BaseModel):
    embeddings: list[list[float]]


class OllamaEmbedder(EmbedderProtocol):
    """Generates dense vectors through the Ollama ``/api/embed`` endpoint."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Create the embedder.

        Args:
            client: HTTP client configured with the Ollama base URL. The caller owns its lifecycle.
            model: Name of an embedding model already pulled into Ollama.
            batch_size: Maximum number of texts sent in a single request.

        Raises:
            ValueError: If batch_size is lower than 1.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._client = client
        self._model = model
        self._batch_size = batch_size

    async def embed_text(self, text: str) -> list[float]:
        """Generate the embedding of a single text.

        Args:
            text: Text content to embed.

        Returns:
            Dense vector representing the text.

        Raises:
            EmbeddingError: If the request fails or the response is unusable.
        """
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for many texts, splitting them into bounded requests.

        Args:
            texts: Text strings to embed.

        Returns:
            Dense vectors in the same order as the input texts.

        Raises:
            EmbeddingError: If a request fails or a response is unusable.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._request_embeddings(texts[start : start + self._batch_size]))
        return vectors

    async def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "/api/embed", json={"model": self._model, "input": texts, "truncate": False}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama request failed: {exc!r}") from exc

        try:
            embeddings = _EmbedResponse.model_validate_json(response.content).embeddings
        except ValidationError as exc:
            raise EmbeddingError("Ollama returned a malformed embedding response") from exc

        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
            )
        return embeddings
