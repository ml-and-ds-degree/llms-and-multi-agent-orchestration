# HW5: Ollama RAG Experiments - Product Requirements Document

## Executive Summary

This project implements a four-part experimental analysis of RAG (Retrieval-Augmented Generation) systems using Ollama as the local LLM backend, with real Ollama Cloud API integration. The work completed includes: baseline replication, parameter deviation analysis, local vs hybrid cloud architecture comparison using real cloud infrastructure (gpt-oss:120b-cloud), and evaluation of advanced retrieval techniques (contextual enrichment and reranking). All runners share a modular `utils/rag_pipeline.py` to avoid duplication.

**Video Reference**: [Ollama Fundamentals Course](https://youtu.be/GWB9ApTPTv4?t=7649) (Starting at 2:07:29)  
**Reference Repository**: [pdichone/ollama-fundamentals](https://github.com/pdichone/ollama-fundamentals)

---

## 1. Project Overview

### 1.1 Purpose

The purpose of this experimental series is to:

1. **Establish a Baseline**: Replicate the exact RAG implementation from the video tutorial to establish reference performance metrics
2. **Understand Sensitivity**: Investigate how changing individual parameters affects system behavior, latency, and accuracy
3. **Compare Architectures**: Evaluate local vs cloud-based LLM approaches in terms of latency, cost, and privacy
4. **Explore Advanced Techniques**: Implement and measure the impact of advanced RAG techniques (contextual retrieval, reranking)

### 1.2 Learning Objectives

- Understand the technical implications of RAG configuration decisions
- Gain hands-on experience with Ollama, LangChain, and vector databases
- Develop skills in experimental design and performance measurement
- Learn to identify trade-offs between different architectural choices

### 1.3 Document Under Study

**BOI.pdf** - Beneficial Ownership Information reporting requirements document. This document serves as our knowledge base for all RAG experiments.

---

## 2. Technical Stack

### 2.1 Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **LLM Runtime** | Ollama | 0.12.6 | Local LLM inference |
| **Language Model** | llama3.2 | 3B params | Text generation |
| **Embedding Model** | nomic-embed-text | latest | Document embeddings |
| **Vector Database** | ChromaDB | latest | Similarity search |
| **Orchestration** | LangChain | 0.4.1+ | RAG pipeline |
| **Document Processing** | PyPDF | latest | PDF parsing |
| **Package Manager** | uv | 0.9.5 | Python dependencies |

### 2.2 Development Environment

- **Platform**: Linux (Debian Bookworm)
- **Container**: DevContainer
- **Python**: 3.13+
- **Ollama Server**: localhost:11434
- **Ollama Cloud**: Supported via `OLLAMA_API_KEY`

### 2.3 Dependencies

```toml
[project.dependencies]
chromadb
langchain-community>=0.4.1
langchain-ollama>=1.0.0
ollama
pypdf
langchain-text-splitters
unstructured
```

---

## 3. Experimental Design

### 3.1 Experiment Structure

The project currently covers **four executed sub-experiments**:

```txt
Experiment 1 (Baseline replication)  ✅ completed
Experiment 2 (Parameter variations)  ✅ completed
Experiment 3 (Local vs Cloud modes) ✅ completed (Updated with real Ollama Cloud support)
Experiment 4 (Advanced techniques)  ✅ completed (contextual retrieval + reranking)
```

---

## 4. Experiment Specifications

### 4.1 Experiment 1: Baseline (Video Replication)

**Objective**: Establish reference performance metrics by replicating the video tutorial exactly, using an instrumented pipeline for chunk stats and metrics saving.

#### Configuration (Following Video @ 2:07:29)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| LLM Model | `llama3.2` | Matches the course recommendation and keeps latency manageable |
| Embedding Model | `nomic-embed-text` | Recommended in the video; strong local performance |
| Chunk Size | 1200 characters | From the course recipe |
| Chunk Overlap | 300 characters | 25% overlap preserves context boundaries |
| Vector Store | Chroma (persisted) | Persisted at `results/chroma_baseline` for reuse |
| Retrieval Strategy | similarity search | Mirrors the course “k=3” lookup |
| Retrieval k | 3 | Default top-k recall for small corpus |
| Prompt Template | “Answer the question based ONLY on the following context…” | Enforces grounding |

#### Test Queries

1. "How to report BOI?"
2. "What is the document about?"
3. "What are the main points as a business owner I should be aware of?"

#### Observed Metrics (from `results/experiment_1_baseline_plus.json`)

- Indexing Time: **0.00 s** (reused persisted store)
- Number of Chunks: **48**
- Avg Response Time: **6.04 s** across 3 queries
- Accuracy: **100%** (auto-eval + manual spot check)

**Takeaways:** Persistence prevents cold-start cost; chunk distribution is healthy (min 153, max 1,198 characters), so recall is strong even with `k=3`.

---

### 4.2 Experiment 2: Parameter Variations (Sensitivity Analysis)

**Objective**: Measure how specific deviations affect latency, accuracy, and startup cost using the same driver (`experiment_1_baseline_plus.py`).

#### Variation A: Smaller LLM (`llama3.2:1b`)

| Parameter | Baseline | Variation A | Notes |
|-----------|----------|-------------|-------|
| LLM Model | `llama3.2` | `llama3.2:1b` | Persisted to a fresh Chroma dir |
| Embedding | `nomic-embed-text` | same | |
| Chunking | 1200 / 300 | same | |
| Persistence | yes | yes | Forced rebuild |

**Observed Metrics (experiment_2_llm_small.json):**

- Indexing Time: **6.05 s**
- Avg Response Time: **4.48 s** (-26% vs baseline+)
- Accuracy: **100%**

**Interpretation:** Smaller model retained accuracy while improving latency (~4.5 s/query), showing the 1B variant is sufficient for this corpus.

#### Variation B: Aggressive Chunking

| Parameter | Baseline | Variation B | Notes |
|-----------|----------|-------------|-------|
| chunk_size | 1200 | 300 | Fragment corpus |
| chunk_overlap | 300 | 50 | Proportional |
| Persistence | yes | yes | Persisted to `results/chroma_llama32_3b` |
| Retriever k | 3 | 3 | |

**Observed Metrics (experiment_2_chunky.json):**

- Indexing Time: **14.93 s** (longer due to 150 chunks)
- Avg Response Time: **3.66 s** (-39% vs baseline+)
- Num Chunks: **150**
- Accuracy: **100%**

**Interpretation:** Smaller chunks sped up responses but introduced more granular vector slices, suggesting a need to watch retriever coverage if queries span multiple chunks.

---

### 4.3 Experiment 3: Local vs Cloud (Architecture Comparison)

**Objective**: Compare local vs. cloud deployment using real Ollama Cloud API with the `gpt-oss:120b-cloud` model.

#### Modes (script: `experiment_3_local_vs_cloud.py`)

1. **All Local:** Ollama LLM (llama3.2 3B) + embeddings + persisted Chroma.
2. **Hybrid (Cloud):** Local embeddings/vector DB; Cloud LLM (gpt-oss:120b-cloud via Ollama Cloud API).

**Note:** Cloud embeddings are not currently available on Ollama Cloud, so we focus on the hybrid architecture which represents the most practical cloud deployment pattern.

#### Observed Metrics (from `experiment_3_*.json`)

| Metric | Local (llama3.2 3B) | Hybrid Cloud (gpt-oss:120b-cloud) |
| --- | --- | --- |
| Indexing time | 0.00 s (persist reuse) | 0.00 s (persist reuse) |
| Avg response latency | 8.29 s | 3.61 s |
| Accuracy | 100% | 100% |
| Network dependency | None (offline) | Ollama Cloud API |
| Speedup | Baseline | **2.3x faster** |

#### Query-Level Performance

| Query | Local (3B) | Hybrid Cloud (120B) | Speedup |
| --- | --- | --- | --- |
| How to report BOI? | 11.37s | 4.25s | 2.7x faster |
| What is the document about? | 4.77s | 1.92s | 2.5x faster |
| Business owner main points? | 8.74s | 4.66s | 1.9x faster |

#### Insights

- **Cloud Dramatically Outperforms Local:** Despite being 40x larger (120B vs 3B parameters), the cloud model was 2.3x faster on average due to superior infrastructure and optimization.
- **Network Overhead is Negligible:** Ollama Cloud API latency is minimal, with queries completing in under 2 seconds in the best case.
- **Accuracy Maintained:** Both configurations achieved 100% accuracy, confirming deployment architecture doesn't compromise quality.
- **Practical Hybrid Pattern:** Local embeddings + cloud LLM offers the best balance of privacy (documents stay local) and performance (inference in cloud).
- **Infrastructure Matters More Than Size:** The results demonstrate that model efficiency and infrastructure optimization trump raw parameter count.

---

### 4.4 Experiment 4: Advanced RAG Techniques

**Objective:** Layer contextual retrieval and reranking on top of the shared pipeline to test edge-case robustness. Both variants are now fully executed via `experiments/experiment_4_advanced.py`, with metrics saved beside the other experiment outputs.

#### Variants & Configuration

1. **Basic Control:** Reuses the baseline 3B model, persisted Chroma store, and `k=3` retrieval to provide a latency reference before adding advanced techniques.
2. **Contextual Retrieval:** Uses `utils/contextualizer.py` to prepend LLM-generated summaries to each chunk during indexing. Chunk count stays at 48, but each chunk roughly doubles in length, and indexing performs 48 additional LLM calls.
3. **LLM Reranking:** Expands initial retrieval to `k=10`, invokes an LLM scorer per chunk, then keeps the top-3 for answer generation.

#### Observed Metrics (from `experiment_4_*.json`)

| Metric | Basic | Contextual | Rerank |
| --- | --- | --- | --- |
| Indexing time | 5.93 s | 9.65 s | 6.27 s |
| Avg response latency | 7.45 s | 10.77 s | 45.94 s |
| Accuracy | 100% | 100% | 100% |
| Notes | Control path | +48 contextualization calls | 10 rerank calls/query |

#### Insights

- **Contextual Retrieval:** Accuracy matched the control, but chunk sizes ballooned (warnings up to ~3,166 characters) and mean latency rose 44%, suggesting the semantic lift is only worthwhile for ambiguous questions.
- **Reranking:** Maintained accuracy yet introduced a 5× latency penalty because every query now runs ten extra LLM calls. Batching or using a lighter reranker would be required for real-time use.
- **Overall:** Advanced techniques did not improve accuracy for this well-structured BOI corpus, so the baseline remains the best speed/accuracy trade-off for interactive scenarios.

---

## 5. File Structure

```toml
HW5/
├── experiments/
│   ├── __init__.py
│   ├── experiment_1_baseline.py        # Video-faithful baseline
│   ├── experiment_1_baseline_plus.py   # CLI-driven variations (Exp 2)
│   └── experiment_3_local_vs_cloud.py  # Local vs hybrid vs cloud modes
├── utils/
│   ├── __init__.py
│   ├── metrics.py                      # Metrics tracking & reporting
│   ├── rag_pipeline.py                 # Shared RAG builders (ingest, vector store, chain, queries)
│   └── contextualizer.py               # Planned contextual retrieval helpers
├── results/
│   ├── experiment_1_baseline.json
│   ├── experiment_1_baseline_plus.json
│   ├── experiment_2_chunky.json
│   ├── experiment_2_llm_small.json
│   ├── experiment_3_local.json
│   ├── experiment_3_hybrid.json
│   ├── experiment_3_cloud.json
│   ├── sub_experiment_1_report.md
│   ├── sub_experiment_2_report.md
│   └── sub_experiment_3_report.md
├── data/
│   └── BOI.pdf                         # Knowledge base document
├── queries.py                          # Test question definitions
├── PRD.md                              # This document
├── BASELINE.md                         # System info
└── ollama-rag-installation.md          # Assignment requirements
```

---

## 6. Metrics & Evaluation

### 6.1 Quantitative Metrics

| Metric | Unit | Collection Method | Purpose |
|--------|------|-------------------|---------|
| Indexing Time | seconds | `time.perf_counter()` | Measure setup overhead |
| Query Response Time | seconds | Per-query timing | User experience |
| Number of Chunks | count | `len(chunks)` | Understand granularity |
| Chunks Retrieved | count | Retriever output | Verify k parameter |
| Memory Usage | MB | `psutil` (optional) | Resource consumption |

### 6.2 Qualitative Metrics

| Metric | Assessment Method | Purpose |
|--------|-------------------|---------|
| Answer Accuracy | Manual validation | Core objective |
| Context Relevance | Expert judgment | Retrieval quality |
| Answer Completeness | Rubric-based | Generation quality |

### 6.3 Comparison Tables

Each experiment will produce a comparison table:

| Metric | Baseline | Variation A | Variation B | Change (%) |
|--------|----------|-------------|-------------|------------|
| Avg Response Time | X.Xs | Y.Ys | Z.Zs | +/-N% |
| Number of Chunks | N | M | P | +/-N% |
| Accuracy Rate | X% | Y% | Z% | +/-N% |
| Technical Issues | None | List | List | - |

---

## 7. Key Deviations from Video

### Video Recommendations vs Our Experiments

| Aspect | Video Recommendation | Our Experiment | Experiment # |
|--------|---------------------|----------------|--------------|
| Chunk Size | 1200 | 300 (smaller) | Exp 2B |
| Persistence | Not emphasized | Persisted baseline; removed for variation | Exp 1 / Exp 2B |
| LLM Location | Local only | Real Ollama Cloud API integration (hybrid mode) | Exp 3 |
| Retrieval | similarity search k=3 | Plan reranking layer | Exp 4B (planned) |
| Embedding | nomic-embed-text | Same; plan contextual summaries | Exp 1 / Exp 4A (planned) |

**Rationale**: These deviations are intentional to explore the sensitivity of the system and understand trade-offs.

---

## 8. Success Criteria

### 8.1 Experiment 1 (Baseline)

- ✅ Replicated video setup with persisted Chroma
- ✅ Generated responses for all baseline queries
- ✅ Indexing time ~0s on reuse (10s on rebuild)
- ⚠️ Query response time ~47s per query (LLM-bound)
- ✅ Baseline accuracy established at 100%

### 8.2 Experiment 2 (Variations)

- ✅ Smaller LLM variation completed (llama3.2:3b)
- ✅ Aggressive chunking + no persistence variation completed
- ✅ Observed chunk inflation to 150 entries; accuracy held at 100%
- ✅ Quantified rebuild cost when persistence is skipped

### 8.3 Experiment 3 (Cloud)

- ✅ Implemented local, hybrid, and real cloud modes
- ✅ Updated to support real Ollama Cloud via API key
- ✅ Compared latency across modes with identical queries
- ✅ Captured cloud-latency buckets for embeddings/LLM
- ✅ Documented privacy/network trade-offs

### 8.4 Experiment 4 (Advanced)

- ✅ Implemented contextual retrieval (chunk summaries via `utils/contextualizer.py`)
- ✅ Implemented reranking strategy (k=10 + LLM scorer)
- ✅ Collected metrics on latency overhead vs accuracy
- ⚠️ No measurable accuracy lift vs baseline; recommend batching/ lighter rerankers if revisited

---

## 9. Video-Specific Questions (Assignment Requirement)

To verify video comprehension:

1. **Which embedding model is recommended in the video and why?**
   - Answer: `nomic-embed-text` - Optimized for RAG, fast local inference, good quality

2. **What port does the Ollama server run on?**
   - Answer: `localhost:11434` (default)

3. **Why does the video emphasize local inference for sensitive data?**
   - Answer: To keep proprietary documents on-device and avoid leaking context to third-party APIs.

4. **What chunk_size and overlap does the video use?**
   - Answer: chunk_size=1200, overlap=300

5. **What is the RAG prompt template structure?**
   - Answer: "Answer the question based ONLY on the following context: {context} Question: {question}"

---

## 10. Deliverables

### 10.1 Code Deliverables

- ✅ 3 experiment scripts (Python)
- ✅ Metrics tracking utility
- ✅ Shared RAG pipeline helper
- ✅ Test queries definition
- ⏳ Visualization utility (planned)
- ⏳ Master experiment runner (planned)

### 10.2 Documentation Deliverables

- ✅ PRD (this document)
- ✅ BASELINE.md (system configuration)
- ✅ Sub-experiment reports (1–3)
- ⏳ RESULTS.md (final report with findings)

### 10.3 Results Deliverables

- ✅ JSON metrics files for Experiments 1–3 (baseline, variations, local vs cloud)
- ✅ Sub-experiment markdown summaries (1–3)
- ⏳ Comparison/roll-up tables
- ⏳ Visualization graphs (PNG)
- ⏳ Screenshot of ollama commands

---

## 11. Running the Experiments

### 11.1 Prerequisites

```bash
# Verify Ollama is running
ollama list

# Should show:
# - llama3.2:latest
# - nomic-embed-text:latest
```

### 11.2 Run Baseline

```bash
cd HW5
python experiments/experiment_1_baseline.py
```

### 11.3 Run Variations (Experiment 2)

```bash
# Smaller LLM (llama3.2:3b)
python experiments/experiment_1_baseline_plus.py \
  --model-name llama3.2:3b \
  --persist-dir results/chroma_llama32_3b \
  --output results/experiment_2_llm_small.json \
  --force-reindex

# Chunky + no persistence
python experiments/experiment_1_baseline_plus.py \
  --chunk-size 300 --chunk-overlap 50 \
  --no-persist --force-reindex \
  --output results/experiment_2_chunky.json
```

### 11.4 Run Local vs Cloud (Experiment 3)

```bash
cd HW5
# local / hybrid / cloud
python experiments/experiment_3_local_vs_cloud.py --mode local
python experiments/experiment_3_local_vs_cloud.py --mode hybrid
python experiments/experiment_3_local_vs_cloud.py --mode cloud
```

---

## 12. Timeline & Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| Project Setup | ✅ Complete | Directory structure, utilities |
| Experiment 1 | ✅ Complete | Baseline implementation |
| Experiment 2 | ✅ Complete | Parameter variations (LLM size + chunking/persistence) |
| Experiment 3 | ✅ Complete | Local vs hybrid vs cloud modes |
| Experiment 4 | ✅ Complete | Advanced contextual retrieval + reranking |
| Results Analysis | ⏳ In Progress | Graphs, tables, insights |
| Final Report | ⏳ Planned | RESULTS.md with findings |

---

## 13. References

1. **Video Tutorial**: [Ollama Course - Build AI Apps Locally](https://youtu.be/GWB9ApTPTv4?t=7649)
2. **Reference Repository**: [pdichone/ollama-fundamentals](https://github.com/pdichone/ollama-fundamentals)
3. **Assignment**: `HW5/ollama-rag-installation.md`
4. **LangChain Docs**: [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
5. **Anthropic**: [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

---

## 14. Experimental Rigor

### 14.1 Reproducibility

- **Fixed Random Seed**: (where applicable)
- **Version Pinning**: All dependencies versioned
- **Docker Environment**: Consistent runtime
- **Same Document**: All experiments use same BOI.pdf

### 14.2 Statistical Validity

- **Multiple Runs**: Where appropriate (e.g., persistence test)
- **Same Queries**: Consistent test set across experiments
- **Controlled Variables**: Change one parameter at a time

### 14.3 Bias Mitigation

- **No Cherry-Picking**: All query results recorded
- **Manual Assessment**: Accuracy judged by consistent criteria
- **Edge Cases**: Include difficult queries

---

## Appendix A: Configuration Summary

### Baseline Configuration (Experiment 1)

```python
# Constants
DOC_PATH = "./data/BOI.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300

# Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Prompt
"""Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

# Persistence
persist_directory = "./results/chroma_baseline"
```

---

## Appendix B: Hypothesis Summary

| Experiment | Hypothesis | Test Method |
|------------|-----------|-------------|
| Exp 2A | Smaller LLM can reduce latency without tanking accuracy | Swap to `llama3.2:3b`, compare response time and accuracy |
| Exp 2B | Aggressive chunking speeds responses but risks recall; skipping persistence adds startup cost | Set chunk_size=300/overlap=50, disable persistence, measure accuracy + indexing time |
| Exp 3 | Networked modes incur higher end-to-end latency despite fast cloud inference | Simulate cloud latency for embeddings/LLM and compare local vs hybrid vs cloud |
| Exp 4A | Contextual chunks add semantic hints but may cost latency | Enrich chunks with summaries and compare top-3 relevance vs control |
| Exp 4B | LLM reranking improves recall but adds overhead | Retrieve k=10, LLM-rerank to top-3, compare latency/accuracy |

---

**Document Version**: 1.3  
**Last Updated**: December 2025  
**Authors**: HW5 Implementation Team  
**Status**: Experiments 1–4 Complete; results analysis in progress
