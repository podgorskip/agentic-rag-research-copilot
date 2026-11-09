# ADR-003: Embedding Backend and Ingestion Pipeline

**Date**: 2026-11-09  
**Status**: Proposed  

## Context

Milestone 2 needs chunks embedded before they can be upserted into Qdrant, and a single entry point that runs Load -> Chunk -> Embed -> Upsert. The embedding model was left open: a local model or an external API.

## Decision

### 1. Embed through Ollama's HTTP API

`OllamaEmbedder` calls `POST /api/embed` with `httpx`, defaulting to `nomic-embed-text` (768 dimensions). Ollama is already the project's local model runtime, so embedding adds no `torch` or ONNX dependency to the application, and CI needs no model because tests fake the HTTP layer. `EmbedderProtocol` keeps the backend swappable.

### 2. Orchestrate against protocols, wire adapters in one place

`IngestionPipeline` depends only on domain protocols and takes the existing `IngestRequest` / `IngestResponse` models, so a FastAPI route can reuse it unchanged. Concrete adapters are constructed only in `ingestion/cli.py`, configured through `RAG_*` environment variables.

### 3. Fail per source, not per batch

Every stage failure becomes an `IngestionError` naming the source and stage. The CLI reports it and moves on to the next source. It exits with 0 on full success, 1 if any source failed and 2 on invalid configuration. There are no automatic retries: chunk IDs are deterministic, so re-running a failed source is safe.

## Consequences

- Ingestion requires a running Ollama with the embedding model pulled.
- The Qdrant collection takes its vector size from the first upsert, so changing `RAG_EMBEDDING_MODEL` to one with a different dimension requires a new collection (`RAG_QDRANT_COLLECTION`).
- A chunk longer than the model's context fails the embed stage instead of being silently truncated, so `RAG_CHUNK_SIZE` must stay within the model's limit.
- Sources are processed sequentially, which is simple but slower for large batches.
