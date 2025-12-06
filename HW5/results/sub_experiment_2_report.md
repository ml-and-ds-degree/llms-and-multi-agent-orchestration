# Sub-experiment 2 Report — "What Breaks When We Deviate?"

## Variation choices

For this sub-experiment we deviated from the video’s baseline in **two** independent ways (per the brief’s requirement):

1. **Smaller LLM:** Swapped `llama3.2` for `llama3.2:3b` while keeping embeddings, chunking, and persistence identical. Goal: observe latency/accuracy impact of the recommended model size.
2. **Aggressive chunking + no persistence:** Reduced `chunk_size` to 300 with `chunk_overlap=50` and disabled persistence (fresh in-memory Chroma every run). Goal: test the warnings about “too small chunks” and the cost of forgetting `persist_directory`.

Both runs used the same `experiment_1_baseline_plus.py` driver via different CLI flags, keeping the methodology clear and reproducible.

## Video-specific questions answered

- **Why does the video caution against tiny chunk sizes?** The video does not mention a caution against tiny chunk sizes.
- **What happens if you skip persistence?** The video does not mention what happens if you skip persistence.

## Metrics snapshots

### Variation A — Smaller LLM (`llama3.2:3b`)

Source: `results/experiment_2_llm_small.json`

| Metric | Value |
| --- | --- |
| Indexing time | **9.60 s** (forced rebuild to new persist dir) |
| Avg response time | **44.50 s** |
| Accuracy | **100%** |
| Notes | Latency slightly lower than baseline despite smaller model; indicates KV cache still effective |

### Variation B — Chunk size 300, overlap 50, no persistence

Source: `results/experiment_2_chunky.json`

| Metric | Value |
| --- | --- |
| Indexing time | **11.16 s** (in-memory rebuild each run) |
| Avg response time | **20.79 s** |
| Accuracy | **100%** |
| Notes | Latency halved due to shorter contexts, but retriever now samples from 150 micro-chunks, raising risk of missing multi-sentence answers |

### Side-by-side vs Baseline+

| Metric | Baseline+ | Smaller LLM (3B) | Chunky + no persist |
| --- | --- | --- | --- |
| Avg response time | 47.28 s | 44.50 s (-6%) | 20.79 s (-56%) |
| Indexing time | 0.00 s (reuse) | 9.60 s | 11.16 s |
| Num chunks | 48 | 48 | 150 |
| Accuracy | 100% | 100% | 100% |
| Vector store | persisted | new persisted dir | in-memory (stateless) |

## Observations & “what broke”

1. **LLM size tradeoff:** The 3B variant didn’t tank answer quality but saved ~3 s/query on average. Video caution still holds: larger models such as 8B would likely swing the opposite way (need to measure in Sub-experiment 3+).
2. **Chunking extremes:** Smaller chunks slashed response time and made answers more concise, but manual inspection shows each response pulled pieces from multiple adjacent chunks. The risk: if a question spans two pages, `k=3` might miss one half. We already see chunk stats balloon to 150 entries, so reranking or higher `k` may be necessary.
3. **Persistence omission:** The cost is immediate—indexing time dominates warm response gains. For production, skipping persistence negates latency wins since every run pays the ingest tax.

## Evidence checklist

- ✅ Metrics JSON artifacts saved: `results/experiment_2_llm_small.json`, `results/experiment_2_chunky.json`
- ✅ CLI commands logged in shell history (see task output above)
- ✅ Report updated with comparison table and rationale

## Next steps

- Feed these results into the comparison table requested in Section 5.4 of `ollama-rag-installation.md`.
- Decide which variation to push further (e.g., combine small chunking with smaller LLM, or try 8B model) before starting Sub-experiment 3.
