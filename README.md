# Agentic RAG Research Copilot

An autonomous multi-source Q&A agent over research papers (arXiv PDFs) and documentation sites.
The system implements a **self-correcting agentic loop**: retrieves context, grades relevance,
rewrites queries when needed, and attributes every answer with precise source citations.

## Stack

| Layer | Component | Default |
|---|---|---|
| Ingestion | PDF/HTML parsing & chunking | `PyMuPDF`, `trafilatura` |
| Retrieval | Hybrid vector + keyword search + reranking | `Qdrant`, `bge-reranker-v2` |
| Agent | Query rewriting, relevance grading, self-correction | Plain Python / `LangGraph` |
| Generation | Answer synthesis with inline citations | `Ollama` (local), `OpenAI` |
| API | Async REST endpoints | `FastAPI` |
| UI | Query interface with streaming | `Streamlit` |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) ≥ 0.4
- Python 3.12 (managed by uv)
- [just](https://github.com/casey/just) (optional, for task runner)
- Qdrant running locally: `docker run -p 6333:6333 qdrant/qdrant`
- Ollama running locally (or set `RAG_LLM_PROVIDER=openai`)

## Quickstart & Development Setup

1. **Install dependencies (including development tools):**
```bash
uv sync --extra dev
```

2. **Start the local Qdrant instance** (requires Docker):
```bash
just qdrant-up 
```
Qdrant is exposed on `http://localhost:6333` (REST) and `localhost:6334` (gRPC). Data is persisted in the
`qdrant_storage` Docker volume. Stop it with `just qdrant-down` and follow its logs with `just qdrant-logs`.

3. **Configure the environment (optional):** copy `config/.env.example` to `config/.env` to override ports, the Qdrant image
   version, or `RAG_QDRANT_URL`.

## Ingesting Documents

Pull the embedding model once, with Qdrant and Ollama running:

```bash
ollama pull nomic-embed-text
```

Then pass any mix of local PDF files and HTML page URLs:

```bash
uv run rag-ingest paper.pdf https://docs.example.org/guide
# or: just ingest paper.pdf https://docs.example.org/guide
```

A source that fails is reported and the rest still run. The exit code is `0` if everything was ingested, `1` if
any source failed and `2` if the configuration is invalid. Settings are listed in `config/.env.example`.

## Code Quality, Typechecking & Testing

The repository enforces strict engineering standards in CI:

* **Strict Typing:** Checked via `pyright` in strict mode.
* **Linting & Formatting:** Managed via `ruff`.
* **Testing & Coverage Gate:** `pytest` with mocked LLM/adapter unit tests and an **80%+ minimum test coverage gate**.

### Manual Commands

```bash
# Auto-fix lint issues and format code
uv run ruff check --fix .
uv run ruff format .

# Run strict static type checking
uv run pyright

# Run unit & integration test suite with 80%+ coverage gate
uv run pytest
```

## Architecture Overview

The codebase follows Domain-Driven Design (`Domain` → `Adapters` → `API`), where external integrations are abstracted behind Python `Protocol` interfaces:

```text
src/agentic_rag_copilot/
├── domain/          # Pure business models (Pydantic) & Protocol interfaces
│   ├── models/      # Entities (Document, Chunk, Query, Citation), AgentState, API schemas
│   └── protocols/   # Abstract contracts (LLM, VectorStore, Embedder, Reranker, Loaders)
├── ingestion/       # PDF & HTML loaders and chunking pipelines
├── retrieval/       # Hybrid search orchestration and cross-encoder reranking
├── agent/           # Self-correcting loop (Rewrite -> Retrieve -> Grade -> Generate)
├── adapters/        # Concrete implementations (Qdrant, Ollama/OpenAI, PyMuPDF)
└── api/             # FastAPI async routes (/ingest, /ask)

```

## Pre-Commit & PR Checklist

Before opening a Pull Request, ensure all items pass locally:

* [ ] **Dependencies synced:** `uv sync --extra dev`
* [ ] **Format & Lint:** `uv run ruff check --fix . && uv run ruff format --check .` passes with zero warnings.
* [ ] **Strict Types:** `uv run pyright` reports `0 errors, 0 warnings`.
* [ ] **Unit Tests & Coverage:** `uv run pytest` passes and total coverage remains **≥ 80%**.
* [ ] **Architecture Boundaries:** Domain modules (`src/agentic_rag_copilot/domain/`) do not import any third-party adapter SDKs.
* [ ] **Documentation:** Any significant architectural trade-offs are documented under `docs/adr/`.
