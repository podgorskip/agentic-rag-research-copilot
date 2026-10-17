from typing import TypeVar

from pydantic import BaseModel

from agentic_rag_copilot.domain.models.entities import Chunk, Document, Query, SourceType
from agentic_rag_copilot.domain.models.state import GradeResult
from agentic_rag_copilot.domain.protocols.ingestion import ChunkerProtocol, DocumentLoaderProtocol
from agentic_rag_copilot.domain.protocols.llm import LLMProtocol
from agentic_rag_copilot.domain.protocols.retrieval import (
    EmbedderProtocol,
    RerankerProtocol,
    VectorStoreProtocol,
)

T = TypeVar("T", bound=BaseModel)


class DummyLoader:
    async def load(self, source_uri: str) -> Document:
        return Document(
            id="d1",
            source_type=SourceType.PDF,
            source_uri=source_uri,
            raw_content="Dummy content",
        )


class DummyChunker:
    def chunk(self, document: Document) -> list[Chunk]:
        return [Chunk(id="c1", document_id=document.id, text=document.raw_content)]


class DummyEmbedder:
    async def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class DummyVectorStore:
    async def upsert(self, chunks: list[Chunk]) -> None:
        return None

    async def hybrid_search(self, query: Query, top_k: int = 10) -> list[Chunk]:
        return [Chunk(id="c1", document_id="d1", text="Result", score=0.9)]


class DummyReranker:
    async def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        return chunks[:top_k]


class DummyLLM:
    async def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        return f"Echo: {prompt}"

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        # Dynamically validate a dummy payload for GradeResult in tests
        return response_model.model_validate({"is_relevant": True, "reasoning": "Mocked reasoning"})


async def test_ingestion_protocols_compliance() -> None:
    loader = DummyLoader()
    chunker = DummyChunker()

    assert isinstance(loader, DocumentLoaderProtocol)
    assert isinstance(chunker, ChunkerProtocol)

    doc = await loader.load("file://test.pdf")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].document_id == "d1"


async def test_retrieval_protocols_compliance() -> None:
    embedder = DummyEmbedder()
    store = DummyVectorStore()
    reranker = DummyReranker()

    assert isinstance(embedder, EmbedderProtocol)
    assert isinstance(store, VectorStoreProtocol)
    assert isinstance(reranker, RerankerProtocol)

    vec = await embedder.embed_text("hello")
    batch_vecs = await embedder.embed_batch(["hello", "world"])
    assert len(vec) == 2
    assert len(batch_vecs) == 2

    q = Query(text="hello", embedding=vec)
    await store.upsert([])
    results = await store.hybrid_search(q, top_k=5)
    reranked = await reranker.rerank(q.text, results, top_k=1)
    assert len(reranked) == 1


async def test_llm_protocol_compliance() -> None:
    llm = DummyLLM()
    assert isinstance(llm, LLMProtocol)

    text_resp = await llm.generate_text("Rewrite query")
    assert text_resp == "Echo: Rewrite query"

    structured_resp = await llm.generate_structured("Grade chunks", GradeResult)
    assert isinstance(structured_resp, GradeResult)
    assert structured_resp.is_relevant is True
