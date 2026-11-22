# ADR-005: Self-Correcting Agentic Loop (Retrieve → Grade → Rewrite → Generate)

**Date**: 2026-11-22  
**Status**: Accepted  

## Context

Milestone 3 ([ADR-004](ADR-004-hybrid-search-and-reranking.md)) delivered single-shot two-stage retrieval. Its validation report identified two failure modes that a single query cannot fix:

1. **Paraphrase sensitivity**: the cross-encoder demotes relevant chunks when the user's wording (e.g. *"hot-swapping"*) does not appear in the chunk text.
2. **Multi-part questions**: questions combining facts from separate sections (e.g. *document encoder frozen vs. fine-tuned*) are not answered by any single 1000-character chunk.

Milestone 4 adds an agent on top of `RetrievalService` that judges whether the retrieved context is sufficient, reformulates the query when it is not, and only then generates a cited answer. The building blocks were delivered by separate issues against `LLMProtocol`: `RelevanceGrader` (structured `GradeResult`), `QueryRewriter` (free-text query), and `AnswerSynthesizer` (structured `SynthesizedAnswer`). This ADR records how they are combined.

## Decision

### 1. Explicit, hand-written state machine in `AgenticLoopService`

`AgenticLoopService.ask(AskRequest) -> AskResponse` (`agent/_loop.py`) drives a plain `while` loop over a mutable `AgentState`:

```text
            ┌──────────────────────────────────────────────┐
            ▼                                              │
  Retrieve(current_query) → merge chunks → Grade ──insufficient & loop_count < max──→ Rewrite
                                             │                       (current_query, query_history, loop_count += 1)
                                 sufficient OR loop_count ≥ max
                                             ▼
                                      Generate (synthesize)
```

* The components (`RelevanceGrader`, `QueryRewriter`, `AnswerSynthesizer`) are **pure with respect to `AgentState`**: they read it and return values. Only the loop mutates state (`is_context_sufficient`, `current_query`, `query_history`, `loop_count`, `retrieved_chunks`, `final_answer`). This keeps each component unit-testable with `MockLLM` and makes the loop the single place where transitions happen.
* No graph framework (LangGraph etc.) is introduced: four states and one back-edge do not justify a new dependency, and the explicit loop is trivially traceable and testable with scripted fakes.
* `query_history` is seeded with the original query and each rewrite is appended, so the rewriter's *"do not repeat any prior query"* instruction covers every query already tried. A repeated rewrite is logged and flagged as `is_repeat` on the `rewrite` span rather than aborting the loop; `max_loop_count` still bounds it.

### 2. Multi-hop chunk accumulation (merge, don't replace)

Each pass merges the newly retrieved chunks into `state.retrieved_chunks`, **de-duplicated by `Chunk.id`, first occurrence wins**. Earlier hops keep their position and later hops only append new chunks. The grader and the synthesizer therefore see the union of all hops, which is what lets a two-part question be answered from chunks found by two different (rewritten) queries.

Each pass retrieves `AskRequest.top_k` chunks (passed as `final_top_k`; `initial_top_k` keeps the `RetrievalService` default). With `top_k=5` and `max_loop_count=3` the context grows to at most 20 chunks (~20k characters).

### 3. Termination and `max_loop_count`

* `AgentSettings.max_loop_count` (`RAG_MAX_LOOP_COUNT`, default `3`, `ge=1`) caps the number of **rewrites**. A request therefore performs between `1` and `max_loop_count + 1` retrieve/grade passes, and `AskResponse.iterations_taken` reports the actual number of passes.
* When the cap is reached without a sufficient grade, the loop **does not raise**: it synthesizes a best-effort answer from everything accumulated. If nothing was retrieved at all, `AnswerSynthesizer` short-circuits (no LLM call) to an explicit "no information found" answer.
* LLM failures (`LLMError`) and contract violations (`ValueError` from an empty rewrite or a citation of an unknown chunk) propagate: they indicate a broken backend or an ungrounded answer, which must not be returned silently.

### 4. Grounded citations

`AnswerSynthesizer` labels each context chunk with its `[n]`, `chunk_id`, `document_id` and `source_uri`, and asks for inline `[n]` markers plus `Citation` entries. After generation it rejects any `chunk_id` not in `retrieved_chunks` (`ValueError`) and overwrites each citation's `document_id`/`source_uri` with the values stored on the cited chunk, so only `quote` is model-authored. To make `source_uri` available, `LangchainTextChunker` now copies `Document.source_uri` into `Chunk.metadata["source_uri"]` (chunks ingested earlier fall back to `document_id`).

### 5. Observability

The whole request is wrapped in an `agentic_loop` Langfuse span (iterations, rewrites, final `query_history`, chunk count). Inside it, `grade`, `rewrite` and `generate` spans follow the `RetrievalService._traced_hybrid_search` / `_traced_rerank` pattern (input, output, `latency_ms` metadata). Retrieval is already traced by `RetrievalService` (`hybrid_search`, `rerank`), so those spans nest under the loop span.

## Consequences

### Positive

* **Self-correction**: insufficient context triggers targeted reformulation instead of a hallucinated answer, directly addressing the Milestone 3 paraphrase and multi-part failure modes.
* **Bounded cost and latency**: worst case is `max_loop_count + 1` retrievals, `max_loop_count + 1` grade calls, `max_loop_count` rewrite calls and one synthesis call.
* **Testability**: the loop is unit-tested with scripted fakes (single-hop, two-rewrite multi-hop, exhaustion), and every step is visible as a Langfuse span.
* **No hallucinated sources**: citations are validated and their identifiers are copied from real chunks.

### Negative / Trade-offs

* **`max_loop_count` trade-off**: a higher cap improves recall on hard multi-hop questions, but each extra pass adds roughly three LLM-bound steps of latency. It also grows the context the grader and synthesizer must read, which raises the risk of prompt truncation on small local models. `3` is a starting point to be tuned from the Milestone 4 validation run.
* **Grading uses `current_query`**: the grader judges the accumulated context against the *current* (possibly narrowed) rewrite, not the original question. A narrow rewrite can be graded sufficient while part of the original question is still uncovered. Grading against `original_query` (or both) is a candidate follow-up if validation shows premature termination.
* **Context growth vs. model context window**: accumulated chunks are not pruned or re-ranked across hops. `OllamaLLM` does not set `num_ctx`, so a model's default context window may silently truncate long grade/synthesis prompts.
* **Strict citation validation can fail requests**: a small model that mangles a 64-character `chunk_id` causes `ValueError` rather than a degraded answer. This is intentional (fail loudly over ungrounded output), but may motivate citing by `[n]` index and mapping indices to chunks server-side.
* **Sequential LLM calls**: grading, rewriting and synthesis are inherently sequential per request; there is no parallel sub-query decomposition yet.
