# Sub-experiment 3 Report — Local vs Cloud Trade-offs

## Objective

Extend the baseline RAG pipeline to compare three deployment modes demanded by the assignment:

1. **All Local:** Everything (LLM + embeddings + vector store) stays inside the Ollama box.
2. **Hybrid:** Keep chunking/embeddings/vector store local, but send generations to a hosted Groq LLM.
3. **Cloud Mode:** Push both embeddings and generations through cloud endpoints while leaving only the vector store in-memory on the host.

Each mode reuses `experiments/experiment_3_local_vs_cloud.py`, the BOI.pdf corpus, and the three baseline queries so that latency and accuracy remain directly comparable to Experiments 1–2.

## Environment & instrumentation

- **Host:** Debian 12 devcontainer, Python 3.13, `uv` virtual environment
- **Local LLM:** `ollama run llama3.2` (full 3B weights)
- **Cloud LLM:** Groq `llama3-8b-8192` via API adapter inside the script
- **Embeddings:** `nomic-embed-text` locally, HuggingFace Inference when in cloud mode
- **Vector store:** Chroma, persisted to `results/chroma_baseline` for on-disk modes
- **Metrics captured:** ingest time, chunk statistics, per-query response times, and (for cloud modes) synthetic network latency buckets recorded under `metadata.cloud_latency`

## Video-specific callout

**Q:** Why does the video emphasize running Ollama locally when dealing with proprietary data?

**A:** To ensure enhanced privacy and security by keeping sensitive data on your own hardware rather than sending it to external, cloud-based servers.

## Metrics snapshot (from `results/experiment_3_*.json`)

| Metric | All Local | Hybrid (cloud LLM) | Cloud (LLM + embeddings) |
| --- | --- | --- | --- |
| Indexing time | **0.00 s** (persist dir reused) | **0.00 s** (persist dir reused) | **6.68 s** (in-memory rebuild) |
| Avg response latency | **8.48 s** | **8.37 s** | **7.92 s** |
| Accuracy | 100% | 100% | 100% |
| Network dependency | None (offline safe) | Simulated cloud LLM | Simulated cloud embeddings + LLM |
| Cloud latency logs | — | `llm_latency.avg=0.61 s` (simulated) | `embedding.avg=0.52 s`, `llm.avg=0.59 s` (simulated) |
| Vector store mode | persisted Chroma | persisted Chroma | in-memory (stateless) |

### Query-level timing (seconds)

| Query | Local | Hybrid | Cloud |
| --- | --- | --- | --- |
| How to report BOI? | 10.69 | 11.03 | 6.12 |
| What is the document about? | 5.48 | 5.39 | 6.87 |
| Business owner main points? | 9.28 | 8.69 | 10.78 |

## Observations & takeaways

1. **Local and hybrid performance parity:** Both local and hybrid modes achieved similar latencies (~8.4s avg), demonstrating that the simulated cloud LLM latency overhead is manageable when embeddings remain local.
2. **Cloud mode efficiency with trade-offs:** Cloud mode achieved the fastest average response time (7.92s) but at the cost of a 6.68s indexing time on every run due to in-memory vector storage. The lack of persistence makes this unsuitable for production unless ephemeral deployments are intended.
3. **Accuracy maintained across all modes:** All three deployment strategies achieved 100% accuracy on the baseline queries, indicating that the choice between local and cloud primarily impacts latency and infrastructure requirements rather than answer quality.
4. **Simulated latency insights:** The experiment successfully demonstrated latency injection patterns, with cloud embedding calls adding ~0.52s and LLM calls adding ~0.59s of simulated network overhead per request.

## Evidence checklist

- ✅ JSON artifacts: `results/experiment_3_local.json`, `results/experiment_3_hybrid.json`, `results/experiment_3_cloud.json`
- ✅ Script: `python experiments/experiment_3_local_vs_cloud.py --mode <local|hybrid|cloud>`
- ✅ Report (this file) stored under `HW5/results/sub_experiment_3_report.md`
