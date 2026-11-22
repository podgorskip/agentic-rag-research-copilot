# Milestone 4 — Agentic Loop Validation Report

> **Status: PENDING REAL-MODEL RUN.** The loop, its unit tests and the end-to-end integration
> tests are implemented, but they have **not yet been run against a real Ollama model**. Every
> results cell below is `TBD` until someone runs the commands in §3 and transcribes the output.
> Nothing in §4–§5 should be read as an observed result until then.

## 1. Overview & Objectives

This report validates the Milestone 4 self-correcting agentic loop (`AgenticLoopService`: Retrieve → Grade → Rewrite → Generate, see [ADR-005](../adrs/ADR-005-agentic-loop.md)) on top of the Milestone 3 retrieval pipeline. It answers three questions:

1. **Rewrite cycles**: how many retrieve/grade passes (`iterations_taken`) and rewrites each query needs.
2. **Gating**: whether the grader correctly *gated* generation, i.e. it reported sufficient context only when the accumulated chunks actually answer the question, and it kept the loop going otherwise.
3. **Grounding & prompts**: whether final citations reference real retrieved chunks, plus any prompt-tuning needed for the grader, rewriter and synthesizer.

---

## 2. Corpus & Pipeline Configuration

### Validation Corpus
- **Paper**: Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* ([arXiv:2005.11401](https://arxiv.org/pdf/2005.11401))
- **Local Cache**: `.cache/validation/2005.11401.pdf`
- **Qdrant Collection**: `validation_lewis_rag_2020` (script) / in-memory `integration_lewis_rag_2020` (pytest)
- **Queries**: the 12 Milestone 3 validation queries (`exact_term`, `conceptual`, `multi_hop`), with focus on the 4 `multi_hop` queries.

### Pipeline Components
| Step | Component | Configuration |
|---|---|---|
| Retrieve | `RetrievalService` (hybrid dense + BM25 → `jinaai/jina-reranker-v2-base-multilingual`) | `initial_top_k=20`, `final_top_k=AskRequest.top_k` |
| Merge | `AgenticLoopService` | union of all hops, de-duplicated by `Chunk.id` |
| Grade | `RelevanceGrader` → `GradeResult` | `RAG_LLM_MODEL` (default `llama3.2`) |
| Rewrite | `QueryRewriter` | same LLM; receives grader `reasoning` and full `query_history` |
| Generate | `AnswerSynthesizer` → `SynthesizedAnswer` | same LLM; citations validated against retrieved chunks |
| Loop cap | `AgentSettings.max_loop_count` | `RAG_MAX_LOOP_COUNT=3` → at most 4 passes |

---

## 3. How to Reproduce

```bash
ollama pull nomic-embed-text && ollama pull llama3.2
just qdrant-up

# Automated end-to-end checks (in-memory Qdrant)
uv run pytest tests/integration/agent -v

# Per-query trace: passes, rewrites, grader verdicts + reasoning, citations
just validate-agent --reingest --output docs/validation/raw_agent_results.md
just validate-agent --category multi_hop --max-loop-count 1 --output docs/validation/raw_agent_results_k1.md
```

`--reingest` is required once so chunks carry `source_uri` metadata (added in this milestone).

---

## 4. Query-by-Query Results (`max_loop_count=3`, `top_k=5`)

| Category | Query | Passes | Rewrites | Grader verdicts | Gated correctly? | Citations grounded? | Notes |
|---|---|---:|---:|---|---|---|---|
| `exact_term` | `RAG-Sequence RAG-Token marginalization equation` | TBD | TBD | TBD | TBD | TBD | |
| `exact_term` | `DPR Dense Passage Retriever BERT-base 100-word Wikipedia` | TBD | TBD | TBD | TBD | TBD | |
| `exact_term` | `MIPS Maximum Inner Product Search FAISS HNSW` | TBD | TBD | TBD | TBD | TBD | |
| `exact_term` | `MSMARCO Jeopardy Question Generation B-1 Q-L ROUGE-L` | TBD | TBD | TBD | TBD | TBD | |
| `conceptual` | `What is the difference between parametric memory and non-parametric memory in RAG?` | TBD | TBD | TBD | TBD | TBD | |
| `conceptual` | `How does RAG-Token differ from RAG-Sequence when generating a target sequence?` | TBD | TBD | TBD | TBD | TBD | |
| `conceptual` | `Why does thorough decoding or fast decoding matter for RAG-Sequence?` | TBD | TBD | TBD | TBD | TBD | |
| `conceptual` | `How does RAG reduce factual hallucinations compared to BART alone?` | TBD | TBD | TBD | TBD | TBD | |
| `multi_hop` | `Which retriever architecture and Wikipedia index size initialize non-parametric memory?` | TBD | TBD | TBD | TBD | TBD | |
| `multi_hop` | `How does RAG perform on FEVER fact verification without supervised retrieval?` | TBD | TBD | TBD | TBD | TBD | |
| `multi_hop` | `What happens during RAG training when the document encoder is frozen vs fine-tuned?` | TBD | TBD | TBD | TBD | TBD | Integration test expects `iterations_taken > 1` at `top_k=2` |
| `multi_hop` | `How does index hot-swapping update world knowledge without retraining the generator?` | TBD | TBD | TBD | TBD | TBD | M3 paraphrase failure case |

**"Gated correctly"** means the final grade was sufficient *and* a manual read confirms the accumulated chunks answer the question, *or* the loop was exhausted on a question the corpus cannot answer.

### Multi-hop rewrite traces
_TBD: paste the per-pass tables (query → verdict → grader reasoning) for at least one `multi_hop` query with `iterations_taken > 1` from `raw_agent_results.md`._

---

## 5. Observations, Gating Quality & Prompt-Tuning Notes

_TBD after the real-model run._ What to check and record, based on known design risks (ADR-005):

1. **Premature sufficiency**: the grader scores against `current_query`, not `original_query`. Check whether a narrow rewrite is graded sufficient while part of the original multi-hop question is still unanswered. If so, pass `original_query` to the grader.
2. **Rewrite cycling**: look for `is_repeat=true` on `rewrite` spans or repeated entries in `query_history`.
3. **Citation failures**: record any `ValueError: LLM cited unknown chunk_id` (reported as **FAILED** by the script). Frequent failures with `llama3.2` would favour citing by `[n]` index and mapping indices server-side.
4. **Context truncation**: at 4 passes × `top_k=5` the prompt can reach ~20k characters. Compare answers at `--top-k 3` vs `5`; if quality drops with more context, set Ollama `num_ctx` explicitly.
5. **`max_loop_count` choice**: compare pass counts and answer quality for `--max-loop-count 1` vs `3` to confirm the default.
