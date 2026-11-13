from agentic_rag_copilot.adapters.retrieval import FastEmbedReranker
from agentic_rag_copilot.domain.models import Chunk


async def test_fastembed_reranker_ranks_relevant_chunk_first() -> None:
    reranker = FastEmbedReranker()
    chunks = [
        Chunk(
            id="distractor-1",
            document_id="doc-1",
            text="Sourdough bread fermentation depends on wild yeast and lactic acid bacteria.",
            metadata={"topic": "baking"},
        ),
        Chunk(
            id="relevant",
            document_id="doc-2",
            text=(
                "Reciprocal Rank Fusion combines dense vector similarity and BM25 lexical "
                "rankings into a single fused relevance list."
            ),
            metadata={"topic": "retrieval"},
        ),
        Chunk(
            id="distractor-2",
            document_id="doc-3",
            text="Photosynthesis converts solar energy into chemical energy inside chloroplasts.",
            metadata={"topic": "biology"},
        ),
    ]

    ranked = await reranker.rerank(
        query="How does Reciprocal Rank Fusion combine BM25 and dense search?",
        chunks=chunks,
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0].id == "relevant"
    assert ranked[0].score is not None
    assert "rerank_score" in ranked[0].metadata
    assert ranked[0].metadata["topic"] == "retrieval"
    assert ranked[0].score >= (ranked[1].score or float("-inf"))
