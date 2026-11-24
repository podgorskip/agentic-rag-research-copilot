"""Configurable ingestion pipeline fake, for use in the API test suite only.

Records every request it receives and can be told to fail, so route tests can exercise both the
success and error responses without loaders, models or a running Qdrant.
"""

from agentic_rag_copilot.domain.models import IngestRequest, IngestResponse
from agentic_rag_copilot.ingestion import IngestionPipeline

from .ingestion import InMemoryVectorStore, MockChunker, MockEmbedder


class MockIngestionPipeline(IngestionPipeline):
    """Returns a configured response, or raises a configured error, for every ingest call."""

    def __init__(
        self,
        response: IngestResponse | None = None,
        *,
        fail_with: Exception | None = None,
    ) -> None:
        super().__init__(
            loaders={},
            chunker=MockChunker(),
            embedder=MockEmbedder(),
            vector_store=InMemoryVectorStore(),
        )
        self.response = response or IngestResponse(document_id="mock-document", chunks_created=1)
        self.fail_with = fail_with
        self.requests: list[IngestRequest] = []

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        self.requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        return self.response
