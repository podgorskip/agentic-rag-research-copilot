# ADR-004: Hybrid Search (Dense + BM25) and Cross-Encoder Reranking

**Date**: 2026-11-13  
**Status**: Accepted  

## Context

Milestone 3 adds two-stage retrieval on top of the ingestion foundation from [ADR-003](ADR-003-embeddings-and-ingestion-pipeline.md):

1. **Hybrid retrieval (`VectorStoreProtocol.hybrid_search`):** Combine dense semantic similarity (`cosine` distance over embeddings from `EmbedderProtocol`) with sparse lexical BM25 matching (`Qdrant/bm25` with IDF term weighting) using Reciprocal Rank Fusion (RRF).
2. **Cross-encoder reranking (`RerankerProtocol.rerank`):** Re-score `(query, chunk.text)` pairs for the top candidates and return the highest-relevance `top_k` chunks.

For the reranker, the original Milestone 3 issue draft suggested `BAAI/bge-reranker-v2-m3` via `sentence-transformers` or `fastembed.TextCrossEncoder`. Inspecting `fastembed==0.5.1` (pinned by `qdrant-client[fastembed]>=1.13,<1.14`) showed that `BAAI/bge-reranker-v2-m3` is not in `fastembed`'s `TextCrossEncoder` registry, whereas `BAAI/bge-reranker-base` and `jinaai/jina-reranker-v2-base-multilingual` are built in. Meanwhile, Ollama has no `/api/rerank` endpoint for cross-encoder models.

## Decision

### 1. Keep FastEmbed / ONNX Runtime as the Local Retrieval Inference Layer

Both BM25 sparse vectorization (`Qdrant/bm25`) and cross-encoder reranking (`FastEmbedReranker`) use `fastembed` (ONNX Runtime) instead of introducing `sentence-transformers` and `torch`.

* **Default reranker model:** `jinaai/jina-reranker-v2-base-multilingual` (configurable via `model_name` in `FastEmbedReranker.__init__`), providing v2 multilingual ranking and a 1,024-token context window with zero additional dependencies.
* **Chunk-size alignment:** `IngestionSettings.chunk_size` defaults to `1000` characters (~200–250 tokens), so `(query, chunk)` pairs fit comfortably within 1,024 tokens without needing `bge-reranker-v2-m3`'s 8,192-token window.
* **Non-blocking execution:** Per [ADR-002](ADR-002-core-models-and-protocols.md), synchronous FastEmbed inference is offloaded via `asyncio.to_thread()` and the model is lazy-loaded on first non-empty call.

### 2. Explicit BM25 Query Vectorization in `QdrantVectorStore.hybrid_search`

In `QdrantVectorStore.upsert()`, `models.Document(text=chunk.text, model="Qdrant/bm25")` is passed directly to Qdrant. However, in `qdrant-client` `1.13.x`, `AsyncQdrantClient._inference_inspector.inspect([FusionQuery(...), prefetch])` returns `False` when `FusionQuery` is paired with a `list[Prefetch]`, causing client-side `Document` inference to be skipped during `query_points`.

To avoid this bug in both in-memory (`:memory:`) and local Qdrant deployments, `QdrantVectorStore.hybrid_search()` computes the query's BM25 `models.SparseVector` explicitly via `fastembed.SparseTextEmbedding("Qdrant/bm25")` inside `asyncio.to_thread()` and passes it to the `"bm25"` `Prefetch`.

### 3. Keep `QdrantVectorStore` under `adapters/vector_store/`

Because `QdrantVectorStore` implements `VectorStoreProtocol` for both ingestion (`upsert`) and retrieval (`hybrid_search`), it remains in `adapters/vector_store/_qdrant_store.py`, while retrieval-only adapters such as `FastEmbedReranker` live in `adapters/retrieval/`.

## Consequences

### Positive

* **Lightweight environment & CI:** Avoids adding `torch`, `transformers`, and `sentence-transformers` (~2–3 GB) to `uv.lock` and CI runners, preserving ADR-003's no-PyTorch dependency footprint.
* **Fast CPU inference:** ONNX Runtime executes both `Qdrant/bm25` and `jinaai/jina-reranker-v2-base-multilingual` efficiently on CPU without requiring a GPU or an extra Docker container (e.g., HuggingFace TEI).
* **Swappable contracts:** `RerankerProtocol` and `FastEmbedReranker(model_name=...)` allow swapping to another FastEmbed model, `sentence-transformers`, or an HTTP rerank service later without touching `RetrievalService` or the agent loop.

### Negative / Trade-offs

* **1,024-token cross-encoder limit:** If `RAG_CHUNK_SIZE` is increased substantially beyond `1000` characters, longer `(query, chunk)` pairs will be truncated by the default model unless swapped for a longer-context backend.
* **CPU-only by default:** Without `onnxruntime-gpu` or a GPU-backed HTTP service, reranking throughput scales with CPU cores rather than CUDA acceleration.
