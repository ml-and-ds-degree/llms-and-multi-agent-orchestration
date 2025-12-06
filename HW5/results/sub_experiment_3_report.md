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
| Indexing time | **8.90 s** (persist dir reused) | **8.69 s** | **11.06 s** (in-memory rebuild) |
| Avg response latency | **43.28 s** | **59.63 s** | **45.85 s** |
| Accuracy | 100% | 100% | 100% |
| Network dependency | None (offline safe) | LLM only (API key + rate limit) | Embeddings + LLM (double dependency) |
| Cloud latency logs | — | `cloud_llm.avg=0.567 s` | `embedding.avg=0.706 s`, `cloud_llm.avg=0.636 s` |
| Vector store mode | persisted Chroma | persisted Chroma | in-memory (stateless) |

### Query-level timing (seconds)

| Query | Local | Hybrid | Cloud |
| --- | --- | --- | --- |
| How to report BOI? | 39.52 | 53.83 | 43.97 |
| What is the document about? | 41.77 | 49.09 | 39.75 |
| Business owner main points? | 48.54 | 75.98 | 53.82 |

## Observations & takeaways

1. **Hybrid overhead dominated by network orchestration:** Even though Groq responded in ~0.57 s, the full path swelled to ~60 s because the pipeline now waits on HTTP setup plus LangChain streaming hooks; local inference avoided that glue code entirely.
2. **Cloud mode regains some latency but at a cold-start cost:** Shipping embeddings to the cloud removed local GPU/CPU contention, trimming per-query latency back near the baseline, but the missing `persist_directory` forced a fresh 11 s indexing tax every run.
3. **Accuracy parity hides privacy trade-offs:** All three modes answered correctly, yet only the local path kept BOI snippets offline. Hybrid/cloud runs require API hygiene (keys, rate limits) that the video warns about for sensitive corpora.
4. **Chunk statistics stay identical:** With chunk_count fixed at 48 and chunk_time ≈0.45–0.58 s, any latency drift clearly stems from network + LLM changes, not preprocessing variance.

## Evidence checklist

- ✅ JSON artifacts: `results/experiment_3_local.json`, `results/experiment_3_hybrid.json`, `results/experiment_3_cloud.json`
- ✅ Script: `python experiments/experiment_3_local_vs_cloud.py --mode <local|hybrid|cloud>`
- ✅ Report (this file) stored under `HW5/results/sub_experiment_3_report.md`
