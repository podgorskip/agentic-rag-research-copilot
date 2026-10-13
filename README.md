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
