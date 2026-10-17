# ADR 002: Core Domain Models, Async Protocols, and Initial Scope Boundaries

* **Date:** 2026-10-17
* * **Status:** Accepted

## Context

The Agentic RAG Research Copilot requires a clean, layered architecture (`Domain` → `Adapters` → `API`) where external technologies—such as PyMuPDF, Qdrant, `bge-reranker-v2`, and Ollama/External LLM APIs—can be swapped without modifying core business logic. Furthermore, the project enforces strict static type checking via `pyright` and requires an 80%+ test coverage gate utilizing mocked adapters.

In establishing the Milestone 1 foundation, we must define the core data structures and abstract interfaces (`Protocol`), while setting clear boundaries on what belongs in the core domain versus what should be deferred to later milestones.

## Decisions

### 1. Strict, Immutable Pydantic V2 Models for Domain Entities

All core data structures (`Document`, `Chunk`, `Citation`) and API contracts (`IngestRequest`, `AskRequest`, `AskResponse`) are implemented using Pydantic V2 with `ConfigDict(strict=True, frozen=True)`.

* **Exception:** `AgentState` uses `frozen=False` to allow controlled state mutation across the self-correcting agent loop (`Rewrite` → `Retrieve` → `Grade` → `Generate`).
* **Rationale:** Runtime type strictness complements static `pyright` analysis, and immutability prevents accidental side effects when passing chunks and documents across adapter boundaries.

### 2. Upfront `Protocol` Interfaces with Generic Structured LLM Outputs

We define `typing.Protocol` interfaces inside `domain/protocols/` for all external dependencies (`DocumentLoaderProtocol`, `ChunkerProtocol`, `EmbedderProtocol`, `VectorStoreProtocol`, `RerankerProtocol`, `LLMProtocol`) prior to concrete adapter implementation.

* For `LLMProtocol`, we use a generic `TypeVar("T", bound=BaseModel)` signature (`generate_structured(prompt, response_model: type[T]) -> T`).
* **Rationale:** Rather than polluting the LLM interface with task-specific methods (e.g., `grade_chunks()`, `extract_citations()`), a generic structured output contract allows the agent layer to request any validated Pydantic schema (`GradeResult`, `SynthesizedAnswer`) while keeping the adapter layer reusable and trivial to mock in unit tests.

### 3. Asynchronous-First Interface Contracts

All I/O-bound protocol methods (loading, embedding, vector search, reranking, and LLM generation) are defined as `async def`. `ChunkerProtocol.chunk()` remains synchronous as it is a pure in-memory string operation.

* **Rationale:** The API layer relies on async REST endpoints via FastAPI. Defining protocols asynchronously from Day 1 prevents event-loop blocking and avoids a breaking interface refactor during Milestone 5. Adapters wrapping synchronous/CPU-bound libraries (e.g., PyMuPDF or local cross-encoders) will offload execution via `asyncio.to_thread()`.

### 4. Scope Boundaries: Stateless Domain & Deferred Streaming

We explicitly exclude user authentication (login/registration), persistent multi-turn session models, and token-streaming entities from the Milestone 1 domain model:

* **Single-Tenant / Stateless API:** The core objective of the system is multi-hop RAG reasoning and Ragas evaluation (faithfulness, recall@k). Authentication and session persistence add database overhead without contributing to retrieval quality. UI session history, if needed, will be managed client-side in Streamlit (`st.session_state`) during Milestone 5.
* **Model Switching via Configuration:** Switching between local Ollama and external APIs will be handled via environment configuration and dependency injection at startup rather than dynamic domain payload routing.
* **Deferred Streaming:** Because Milestones 2–4 focus on ingestion, hybrid retrieval, and the self-correcting loop—and Ragas evaluates complete answers—streaming contracts (`AsyncIterator` / SSE models) are deferred to Milestone 5 (Serving & UI).

## Consequences

### Positive

* **Parallel Development:** Team members can simultaneously build Milestone 2 (Ingestion), Milestone 3 (Retrieval), and Milestone 4 (Agent Loop) against stable, strictly typed contracts.
* **Zero Domain Leakage:** The `domain` package has zero dependencies on `qdrant-client`, `PyMuPDF`, `ollama`, or `fastapi`.
* **High Testability:** Mock adapters (e.g., `FakeLLM`, `InMemoryVectorStore`) can be written in a few lines of code to hit the 80%+ CI coverage requirement without network calls.

### Negative / Trade-offs

* **Async Boilerplate for Sync Libraries:** Developers implementing local CPU-bound adapters in Milestones 2 and 3 must wrap synchronous calls to conform to `async def` signatures.
* **Milestone 5 Extension:** `LLMProtocol` and `api.py` will require a minor, non-breaking extension in Milestone 5 to add a `generate_stream()` method and streaming event payloads for the Streamlit UI.
