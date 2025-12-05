# HW5: Ollama RAG Experiments - Product Requirements Document

## Executive Summary

This project implements a comprehensive experimental analysis of RAG (Retrieval-Augmented Generation) systems using Ollama as the local LLM backend. The experiments systematically explore how different configuration decisions impact RAG performance, following the methodology established in the "Ollama Course - Build AI Apps Locally" video tutorial.

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

The project consists of **4 sub-experiments**, each building on the baseline:

```
Experiment 1 (Baseline)
    ↓
Experiment 2 (Parameter Variations)
    ↓
Experiment 3 (Local vs Cloud)
    ↓
Experiment 4 (Advanced RAG Techniques)
```

---

## 4. Experiment Specifications

### 4.1 Experiment 1: Baseline (Video Replication)

**Objective**: Establish reference performance metrics by replicating the video tutorial exactly.

#### Configuration (Following Video @ 2:07:29)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| LLM Model | llama3.2 (3B) | Recommended in video for balance of speed/quality |
| Embedding Model | nomic-embed-text | Optimized for RAG, fast, local |
| Chunk Size | 1200 characters | Video recommendation for document context |
| Chunk Overlap | 300 characters | 25% overlap to preserve context boundaries |
| Vector Store | Chroma (in-memory) | No persistence in baseline |
| Retrieval Strategy | MultiQueryRetriever | Generates 5 query variations for better recall |
| Retrieval k | 3 (implicit) | Default from MultiQueryRetriever |
| Prompt Template | "Answer based ONLY on..." | Strict context adherence |

#### Test Queries

1. "How to report BOI?" - Direct procedural question
2. "What is the document about?" - General understanding
3. "What are the main points as a business owner I should be aware of?" - Synthesis question

#### Metrics Collected

- **Indexing Time**: Time to chunk + embed + store documents
- **Number of Chunks**: Total chunks created from BOI.pdf
- **Query Response Time**: Per-query latency (avg, min, max)
- **Answer Quality**: Manual assessment (accurate/inaccurate)

#### Expected Outcomes

- Establish baseline latency benchmarks
- Validate that the setup works correctly
- Create reference answers for accuracy comparison

---

### 4.2 Experiment 2: Parameter Variations (Sensitivity Analysis)

**Objective**: Understand how deviating from recommended parameters impacts performance.

#### Variation A: Chunking Parameters

**DEVIATION**: Change from recommended chunk_size=1200 to chunk_size=300

| Parameter | Baseline | Variation A | Rationale |
|-----------|----------|-------------|-----------|
| chunk_size | 1200 | 300 | Test fragmentation impact |
| chunk_overlap | 300 | 50 | Proportional reduction |

**Hypothesis**:

- **Positive**: More precise retrieval (smaller semantic units)
- **Negative**: Lost context, increased chunks (4x), potential information fragmentation

**What to Measure**:

- Number of chunks created (expect ~4x increase)
- Retrieval quality (do answers lose context?)
- Indexing time (expect faster embedding)
- Answer accuracy on synthesis questions (expect degradation)

---

#### Variation B: Vector Store Persistence

**DEVIATION**: Remove `persist_directory` parameter

| Parameter | Baseline | Variation B | Rationale |
|-----------|----------|-------------|-----------|
| persist_directory | None | None (explicit test) | Already baseline, test rebuild cost |
| Test runs | 1 | 3 consecutive | Measure repeated indexing cost |

**Hypothesis**:

- Same accuracy (deterministic with seed)
- Significantly higher startup latency on each run
- Demonstrates production scalability issue

**What to Measure**:

- Indexing time per run (expect consistent ~5-10s)
- Total time for 3 runs vs 1 run with persistence
- User experience degradation

---

### 4.3 Experiment 3: Local vs Cloud (Architecture Comparison)

**Objective**: Compare local Ollama setup against cloud-based LLM alternatives.

#### Mode 1: All Local (Baseline)

```
PDF → Chunks → Ollama Embeddings → ChromaDB → Ollama LLM → Response
[ALL LOCAL]
```

#### Mode 2: Hybrid (Cloud LLM)

```
PDF → Chunks → Ollama Embeddings → ChromaDB → Groq API → Response
[LOCAL PROCESSING] [CLOUD GENERATION]
```

**Cloud Provider**: Groq (free tier)  
**Model**: llama3-8b-8192 (larger model, faster inference)

#### Mode 3: Cloud Mode (Optional)

```
PDF → Chunks → HuggingFace Embeddings → ChromaDB → Groq API → Response
[LOCAL STORAGE] [CLOUD PROCESSING]
```

#### Metrics to Compare

| Metric | Local | Hybrid | Cloud |
|--------|-------|--------|-------|
| Cold Start Latency | ? | ? | ? |
| Query Latency | ? | ? | ? |
| Network Dependency | No | Yes | Yes |
| Cost per Query | $0 | $0 (free tier) | $0 (free tier) |
| Privacy | Full | Partial | Low |
| Offline Capable | Yes | No | No |
| Rate Limits | None | 30 req/min | ? |

#### Expected Insights

- **Local**: Higher latency, full privacy, no network dependency
- **Hybrid**: Potentially lower latency (Groq is fast), network required for generation only
- **Cloud**: Lowest setup friction, highest network dependency

---

### 4.4 Experiment 4: Advanced RAG Techniques

**Objective**: Implement and measure advanced RAG strategies to overcome common failure modes.

#### Technique A: Basic vs Contextual Retrieval

**Basic RAG** (Baseline):

```
Chunk → Embed → Store
```

**Contextual Retrieval** (Enhancement):

```
Chunk → LLM(generate context summary) → Augmented Chunk → Embed → Store
```

**Context Generation Prompt**:

```
"Given this chunk from a document about BOI reporting, generate a 1-2 sentence 
context description that will help with semantic search. Chunk: {chunk_text}"
```

**Test Queries**:

- **Direct**: "What are the filing deadlines?" (should work equally)
- **Implied**: "When do I need to submit this?" (contextual should excel)
- **Ambiguous**: "What are the requirements?" (contextual should disambiguate)

**Measurement**: Retrieval accuracy (% correct chunks in top-3)

---

#### Technique B: Basic Retrieval vs Reranking

**Basic Retrieval**:

```
Query → similarity_search(k=3) → LLM
```

**Reranking**:

```
Query → similarity_search(k=10) → LLM_rerank → Top 3 → LLM
```

**Reranking Prompt**:

```
"Rate how relevant this chunk is to the question on a scale of 0-10.
Question: {question}
Chunk: {chunk}
Rating:"
```

**Hypothesis**: Reranking reduces "missed top rank" failures by casting a wider net initially.

**Measurement**:

- Compare which chunks are selected (Basic vs Reranked)
- Final answer quality improvement
- Total latency cost (expect +20-30% for reranking)

---

## 5. File Structure

```
HW5/
├── experiments/
│   ├── __init__.py
│   ├── experiment_1_baseline.py       # Baseline (video replication)
│   ├── experiment_2_variations.py     # Chunking + Persistence tests
│   ├── experiment_3_cloud.py          # Local vs Cloud comparison
│   └── experiment_4_advanced.py       # Contextual + Reranking
├── utils/
│   ├── __init__.py
│   ├── metrics.py                      # Metrics tracking & reporting
│   └── visualization.py                # Graph generation (planned)
├── results/
│   ├── experiment_1_baseline.json
│   ├── experiment_2_variations.json
│   ├── experiment_3_cloud.json
│   └── experiment_4_advanced.json
├── data/
│   └── BOI.pdf                         # Knowledge base document
├── queries.py                          # Test question definitions
├── run_all_experiments.py              # Master experiment runner (planned)
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
| Chunk Size | 1200 | 300 | Exp 2A |
| Persistence | Not shown (in-memory) | Test rebuild cost | Exp 2B |
| LLM Location | Local only | Add cloud comparison | Exp 3 |
| Retrieval | MultiQueryRetriever | Add reranking | Exp 4B |
| Embedding | Direct chunks | Add contextual | Exp 4A |

**Rationale**: These deviations are intentional to explore the sensitivity of the system and understand trade-offs.

---

## 8. Success Criteria

### 8.1 Experiment 1 (Baseline)

- ✅ Successfully replicate video setup
- ✅ Generate responses for all test queries
- ✅ Indexing time < 30s
- ✅ Query response time < 10s per query
- ✅ Establish baseline accuracy

### 8.2 Experiment 2 (Variations)

- ✅ Demonstrate 4x chunk increase with small chunk_size
- ✅ Show context fragmentation impact on synthesis questions
- ✅ Quantify rebuild cost without persistence

### 8.3 Experiment 3 (Cloud)

- ✅ Successfully integrate Groq API
- ✅ Compare latency (local vs hybrid)
- ✅ Document network dependency impact
- ✅ Measure rate limit behavior

### 8.4 Experiment 4 (Advanced)

- ✅ Implement contextual retrieval
- ✅ Implement reranking strategy
- ✅ Demonstrate improvement on edge cases
- ✅ Quantify latency overhead

---

## 9. Video-Specific Questions (Assignment Requirement)

To verify video comprehension:

1. **Which embedding model is recommended in the video and why?**
   - Answer: `nomic-embed-text` - Optimized for RAG, fast local inference, good quality

2. **What port does the Ollama server run on?**
   - Answer: `localhost:11434` (default)

3. **What is the purpose of MultiQueryRetriever?**
   - Answer: Generate multiple query variations to overcome limitations of distance-based similarity search

4. **What chunk_size and overlap does the video use?**
   - Answer: chunk_size=1200, overlap=300

5. **What is the RAG prompt template structure?**
   - Answer: "Answer the question based ONLY on the following context: {context} Question: {question}"

---

## 10. Deliverables

### 10.1 Code Deliverables

- ✅ 4 experiment scripts (Python)
- ✅ Metrics tracking utility
- ✅ Test queries definition
- ⏳ Visualization utility (planned)
- ⏳ Master experiment runner (planned)

### 10.2 Documentation Deliverables

- ✅ PRD (this document)
- ✅ BASELINE.md (system configuration)
- ⏳ RESULTS.md (final report with findings)

### 10.3 Results Deliverables

- ⏳ 4 JSON metrics files (one per experiment)
- ⏳ Comparison tables (markdown)
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

### 11.3 Run All Experiments

```bash
cd HW5
python run_all_experiments.py  # (planned)
```

---

## 12. Timeline & Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| Project Setup | ✅ Complete | Directory structure, utilities |
| Experiment 1 | ✅ Complete | Baseline implementation |
| Experiment 2 | ⏳ Planned | Parameter variations |
| Experiment 3 | ⏳ Planned | Cloud comparison |
| Experiment 4 | ⏳ Planned | Advanced techniques |
| Results Analysis | ⏳ Planned | Graphs, tables, insights |
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
MultiQueryRetriever(k=5 query variations)

# Prompt
"""Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
```

---

## Appendix B: Hypothesis Summary

| Experiment | Hypothesis | Test Method |
|------------|-----------|-------------|
| Exp 2A | Small chunks hurt synthesis | Compare accuracy on multi-hop questions |
| Exp 2B | No persistence = poor UX | Measure repeated startup time |
| Exp 3 | Cloud = higher latency | Compare response times |
| Exp 4A | Contextual = better edge cases | Retrieval accuracy on implied queries |
| Exp 4B | Reranking = fewer misses | Compare top-k chunks selected |

---

**Document Version**: 1.0  
**Last Updated**: December 5, 2024  
**Authors**: HW5 Implementation Team  
**Status**: Baseline Complete, Experiments 2-4 Planned
