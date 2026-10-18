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
