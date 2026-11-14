# Milestone 3 — Retrieval Quality Validation Report

## 1. Overview & Objectives

This report documents the end-to-end validation of the Milestone 3 retrieval pipeline (`OllamaEmbedder` + `QdrantVectorStore` hybrid search + `FastEmbedReranker` + `RetrievalService`), comparing **Dense-only**, **Hybrid (Dense + BM25)**, and **Hybrid + Reranked** retrieval across 12 curated queries, as well as comparing three parameter configurations.

Validation was executed using [`scripts/validate_retrieval.py`](../../scripts/validate_retrieval.py) (`just validate-retrieval`).

---

## 2. Corpus & Pipeline Configuration

### Validation Corpus
- **Paper**: Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* ([arXiv:2005.11401](https://arxiv.org/pdf/2005.11401))
- **Local Cache**: `.cache/validation/2005.11401.pdf` (downloaded on first run and reused on subsequent runs; gitignored)
- **Qdrant Collection**: `validation_lewis_rag_2020`

### Pipeline Components & End-to-End Flow
1. **Ingestion & Chunking**
   - `PyMuPDFLoader` extracts text from the local cached PDF.
   - `LangchainTextChunker` (`RecursiveCharacterTextSplitter`) splits text with `chunk_size=1000` and `chunk_overlap=200`.
   - `QdrantVectorStore.upsert_chunks` indexes dense vectors (`nomic-embed-text` via `OllamaEmbedder`) alongside sparse BM25 vectors (`Qdrant/bm25` via FastEmbed).
2. **Stage 1 — Broad Hybrid Candidate Retrieval (`QdrantVectorStore.hybrid_search`)**
   - `RetrievalService.retrieve` embeds the query with `OllamaEmbedder` (`nomic-embed-text`) if `query.embedding` is not already populated.
   - `QdrantVectorStore.hybrid_search` executes dual Qdrant `Prefetch` queries (`dense` cosine similarity and `bm25` sparse matching) and fuses them with Reciprocal Rank Fusion (`models.Fusion.RRF`), returning up to `initial_top_k` candidates.
3. **Stage 2 — Cross-Encoder Reranking (`FastEmbedReranker.rerank`)**
   - `FastEmbedReranker` scores all `initial_top_k` candidate chunks against the query text using FastEmbed's `TextCrossEncoder`, attaches `metadata["rerank_score"]` and `score`, and returns the top `final_top_k` chunks sorted by descending relevance.
4. **Observability (`RetrievalService`)**
   - Both `hybrid_search` and `rerank` stages are wrapped in Langfuse spans recording query text, candidate/result counts, latency, and top rerank score.

---

## 3. Parameter Tuning & Configuration Comparison

Three parameter configurations were evaluated against the same `validation_lewis_rag_2020` collection:

| Run | Reranker Model | `initial_top_k` | `final_top_k` | Summary |
|---|---|---:|---:|---|
| **Run 1 (Baseline)** | `jinaai/jina-reranker-v2-base-multilingual` | 20 | 5 | Strong top-5 precision, but `k=5` cut off two key multi-hop/paraphrase chunks (`665dfb39`, `88c791a6`). |
| **Run 2 (BGE Reranker)** | `BAAI/bge-reranker-base` | 30 | 6 | Net regression: frequently over-scored `Table 1` (`78eecf2b`) and demoted exact lexical matches (`1ed1938c`, `f9d847fe`, `c876b23b`, `86c6e228`). |
| **Run 3 (Selected)** | `jinaai/jina-reranker-v2-base-multilingual` | 30 | 6 | **Best overall**: preserves all Run 1 top-5 rankings while `final_top_k=6` recovers `665dfb39` and `88c791a6` into the returned context. |

### Key Parameter Decisions
- **Reranker model (`jinaai/jina-reranker-v2-base-multilingual` vs. `BAAI/bge-reranker-base`)**:
  - `BAAI/bge-reranker-base` improved 2 queries (`b72851f3` on `MIPS ... FAISS HNSW` moved #3 -> #1; `27fdbda6` on RAG-Sequence decoding moved #2 -> #1), **but regressed on 7 queries**:
    - Ranked `Table 1: Open-Domain QA Test Scores` (`78eecf2b`) #2 on the marginalization equation query and #3 on FEVER.
    - Dropped `1ed1938c` (`BERTBASE doc...`, #1 in Dense-only) to #6 (`-1.8225`) on `DPR Dense Passage Retriever BERT-base 100-word Wikipedia`.
    - Dropped `f9d847fe` (`Bleu-1 / Rouge-L score...`) from #3 to #6 (`-1.5466`) on the MSMARCO/Jeopardy metrics query.
    - Demoted `c876b23b` (`non-parametric memory is a dense vector index...`) from #1 to #3 on the parametric vs. non-parametric memory query and from #1 to #2 on the retriever/index-size query.
    - Demoted `86c6e228` (`find evidence for this hypothesis...`) from #2 to #6 (`-0.9668`) on FEVER.
  - Therefore, **`jinaai/jina-reranker-v2-base-multilingual` was retained**.
- **`initial_top_k=30` (up from `20`)**:
  - In Run 3, top-5 results were identical to `initial_top_k=20`, showing `20` was already sufficient for a single-paper corpus, while `30` adds headroom for multi-document corpora without noticeable latency impact.
- **`final_top_k=6` (up from `5`)**:
  - Directly recovered `665dfb39` (`can be easily updated at test time...`) on the `index hot-swapping` query and `88c791a6` (`For FEVER classification...`) on the FEVER query, both of which were cut off at `k=5`.
- **Standalone validation vs. service defaults**:
  - `scripts/validate_retrieval.py` defaults to `initial_top_k=30` and `final_top_k=6`, while `RetrievalService.retrieve` keeps `initial_top_k=20` and `final_top_k=5` until Milestone 4 multi-document synthesis is evaluated.
- **Standalone paper choice**:
  - Evaluating against a single canonical paper (`arXiv:2005.11401`) cached locally in `.cache/validation/` makes validation deterministic, reproducible, and fast to re-run without committing binary PDFs to Git.

---

## 4. Query-by-Query Comparison (Selected Run: `jina-reranker-v2`, `initial_top_k=30`, `final_top_k=6`)

| Category | Query | Dense-only Top 1 | Hybrid Top 1 | Hybrid + Reranked Top 1 (Score) | Key Effect of Hybrid + Reranking |
|---|---|---|---|---|---|
| `exact_term` | `RAG-Sequence RAG-Token marginalization equation` | `aed7d243` (0.7604) | `aed7d243` (1.0000) | `aed7d243` (1.4749) | Surfacing `94d6d41e` (marginal log-likelihood, #4 `0.9731`) and `338909ca` (beam decoder, #5 `0.9711`) into top 5 (Dense had `94d6d41e` at #6 and missed `338909ca`). |
| `exact_term` | `DPR Dense Passage Retriever BERT-base 100-word Wikipedia` | `1ed1938c` (0.7296) | `c876b23b` (0.6667) | `c876b23b` (0.5317) | Pairs `c876b23b` (Wikipedia dense vector index) at #1 and `1ed1938c` (`BERTBASE` doc encoder) at #2 (`0.1492`). |
| `exact_term` | `MIPS Maximum Inner Product Search FAISS HNSW` | `1ed1938c` (0.5948) | `1ed1938c` (0.7000) | `b0c8c880` (0.1053) | Recovers `7a3ad6e2` (`MIPS index using FAISS [23]...`) at #5 (`-0.7510`) from the `top_k=30` candidate pool; absent from Dense and Hybrid top 6. |
| `exact_term` | `MSMARCO Jeopardy Question Generation B-1 Q-L ROUGE-L` | `05dfbad2` (0.7310) | `ea2c9a6b` (0.5833) | `ea2c9a6b` (2.2494) | Promotes `ea2c9a6b` (`Table 4: Human assessments...`) to #1 and `f9d847fe` (`Bleu-1 / Rouge-L score...`, absent in Dense top 6) to #3 (`0.5925`). |
| `conceptual` | `What is the difference between parametric memory and non-parametric memory in RAG?` | `284b0624` (0.7489) | `284b0624` (0.6429) | `c876b23b` (0.9271) | Promotes `c876b23b` (explicit definition of non-parametric memory) from #3 (Dense) / #2 (Hybrid) to #1, ahead of `6 Discussion` (`284b0624`). |
| `conceptual` | `How does RAG-Token differ from RAG-Sequence when generating a target sequence?` | `aed7d243` (0.7499) | `aed7d243` (0.8333) | `3850ad7e` (1.2802) | Promotes token-level generation (`3850ad7e`, #1) and marginal log-likelihood (`94d6d41e`, #2 `1.2563`) ahead of section intro `aed7d243` (#3 `1.0924`). |
| `conceptual` | `Why does thorough decoding or fast decoding matter for RAG-Sequence?` | `27fdbda6` (0.7404) | `27fdbda6` (0.7500) | `338909ca` (0.5949) | Places beam-decoder approximation (`338909ca`, #1) and forward-pass cost (`27fdbda6`, #2 `0.5188`) at the top. |
| `conceptual` | `How does RAG reduce factual hallucinations compared to BART alone?` | `a69e9a63` (0.6923) | `664c7276` (0.7000) | `664c7276` (1.6703) | Pushes `Broader Impact` (`7dab7167`, #2 in Dense) out of top 6; ranks `664c7276` (#1), `80937832` (`in 71% of cases...`, #2 `0.6241`), and `a69e9a63` (#3 `0.4432`). |
| `multi_hop` | `Which retriever architecture and Wikipedia index size initialize non-parametric memory?` | `c876b23b` (0.7981) | `c876b23b` (1.0000) | `c876b23b` (0.7260) | Pairs `c876b23b` (#1) with `1ed1938c` (`BERTBASE` doc encoder, #2 `0.7212`, up from #6 in Hybrid). |
| `multi_hop` | `How does RAG perform on FEVER fact verification without supervised retrieval?` | `ef3bb511` (0.7360) | `5b2565a2` (0.8333) | `5b2565a2` (1.3388) | Promotes `5b2565a2` (#1) and `86c6e228` (`find evidence...`, absent in Dense top 6) to #2 (`1.1422`), while `k=6` retains `88c791a6` (`For FEVER classification...`) at #6 (`0.1953`). |
| `multi_hop` | `What happens during RAG training when the document encoder is frozen vs fine-tuned?` | `278b2719` (0.6814) | `278b2719` (0.7000) | `94d6d41e` (0.3763) | Surfaces training objective (`94d6d41e`, #1), ablation context (`60757e1b`, #2 `-0.1543`), and `RAG-Sequence-Frozen` table (`2ba59c51`, #3 `-0.2238`). |
| `multi_hop` | `How does index hot-swapping update world knowledge without retraining the generator?` | `665dfb39` (0.6440) | `665dfb39` (0.7000) | `284b0624` (0.3963) | Cross-encoder ranks `284b0624` (#1) and `60757e1b` (#2), while `k=6` recovers `665dfb39` (`can be easily updated at test time...`) at #6 (`-0.9828`). |

---

## 5. Observed Improvements & Failure Modes

### Observed Improvements
1. **Exact equation & metric recall**: Hybrid + Reranking surfaced exact mathematical and tabular chunks (`94d6d41e`, `338909ca`, `f9d847fe`, `7a3ad6e2`) that Dense-only missed or ranked outside top 5.
2. **Noise suppression**: Generic sections (`7dab7167` *Broader Impact*, `284b0624` *6 Discussion*) ranked #1–#2 by Dense-only were consistently demoted below specific evidence chunks (`c876b23b`, `80937832`, `664c7276`).
3. **Two-stage recovery**: `7a3ad6e2` (`MIPS index using FAISS [23]`) did not appear in the top 6 of either Dense-only or Hybrid alone, but was recovered from the `initial_top_k=30` pool by the reranker into rank #5.

### Observed Failure Modes (Future Work for Milestone 4)
1. **Lexical sensitivity on paraphrases**: On *"How does index hot-swapping update world knowledge..."*, chunk `665dfb39` (*"can be easily updated at test time..."*) was #1 in both Dense-only and Hybrid, but scored `-0.9828` (#6) under the cross-encoder because the literal phrase *"hot-swapping"* did not appear in the chunk.
2. **Multi-part queries spanning separate sections**: Queries combining distinct sub-facts (`MIPS ... FAISS HNSW`; `document encoder frozen vs fine-tuned`) yield lower/negative cross-encoder logits because no single 1000-character chunk contains all sub-answers. **Milestone 4 agentic query decomposition / rewriting** is the natural place to split multi-hop questions into targeted sub-queries.
3. **Chunk-boundary & table-extraction artifacts**: `chunk_size=1000` with default separators produces some mid-sentence chunk starts (`665dfb39`, `dda83fd1`, `80937832`) and flattened numeric table rows (`2ba59c51`). Increasing `chunk_size` (e.g., `1400–1600`) or using sentence-boundary separators can be explored if multi-document synthesis requires richer standalone passages.
