# Sub-experiment 4 — Contextual Retrieval & Reranking

## Objective

Test two advanced retrieval improvements on the BOI corpus:

1. **Contextual retrieval:** Prepend LLM-generated summaries to each chunk before embedding, aiming to boost relevance for implied/ambiguous queries.
2. **Reranking:** Retrieve a wider candidate set (k=10 by default) and rerank with an LLM scorer, then keep the top-3 before generation.

Both variants reuse the shared pipeline (`utils/rag_pipeline.py`) and the Experiment 4 driver (`experiments/experiment_4_advanced.py`) to stay consistent with prior experiments.

## Metrics captured

| Metric | Basic | Contextual | Rerank |
| --- | --- | --- | --- |
| Indexing time | 5.93 s | 9.65 s | 6.27 s |
| Avg response latency | 7.45 s | 10.77 s | 45.94 s |
| Num chunks | 48 | 48 | 48 |
| Accuracy | 100% | 100% | 100% |
| Notes | Baseline control | +48 LLM calls for contextualization | 10 LLM rerank calls per query |

### Query-level view (baseline queries)

| Question | Basic latency | Contextual latency | Rerank latency | Accurate? |
| --- | --- | --- | --- | --- |
| How to report BOI? | 7.77 s | 8.57 s | 42.52 s | ✅ All |
| What is the document about? | 5.78 s | 7.53 s | 42.91 s | ✅ All |
| What are the main points as a business owner I should be aware of? | 8.79 s | 16.23 s | 52.39 s | ✅ All |

## Interpretation

### Contextual retrieval
- **Indexing overhead:** Adding contextual summaries increased indexing time from 5.93s to 9.65s due to 48 additional LLM calls (one per chunk).
- **Query latency impact:** Average response time increased from 7.45s to 10.77s (+44%), with the longest query taking 16.23s.
- **Accuracy:** Maintained 100% accuracy, suggesting the contextual information didn't harm retrieval quality.
- **Chunk size impact:** Contextualized chunks grew significantly (mean length increased from ~900 to ~2400 characters based on warning logs showing lengths up to 3166).

### Reranking
- **Significant latency cost:** Average response time ballooned to 45.94s (+516% vs basic), as each query required 10 additional LLM scoring calls for reranking.
- **Accuracy maintained:** Despite the latency penalty, accuracy remained at 100%, indicating reranking didn't introduce retrieval errors.
- **Candidate evaluation overhead:** With k=10 candidates retrieved and scored individually, the reranking step dominated query latency (30-40s of the total ~45s).
- **Potential optimization:** Batching reranking calls or reducing candidate count could significantly improve performance.

## Evidence checklist

- ✅ JSON artifacts saved: `results/experiment_4_basic.json`, `results/experiment_4_contextual.json`, `results/experiment_4_rerank.json`
- ✅ Console logs captured for contextual summary generation (48 LLM calls during indexing) and reranking path (10 scoring calls per query)
- ✅ This report updated with the actual metrics from completed runs

## Key takeaways

1. **Contextual retrieval trade-off:** While contextual summaries add meaningful semantic context, the 44% latency increase and larger chunk sizes may not justify the gains for straightforward document Q&A tasks like BOI instructions.

2. **Reranking cost:** The 5x latency penalty from reranking makes it impractical for interactive use without optimization. Potential improvements include:
   - Reducing candidate count from 10 to 5
   - Batching reranking calls
   - Using lighter reranking models
   - Caching reranking scores for similar queries

3. **Accuracy plateau:** All three variants achieved 100% accuracy, suggesting that for this well-structured document, basic retrieval with k=3 is already sufficient.

4. **Production considerations:** For the BOI use case, the basic variant offers the best balance of speed (7.45s avg) and accuracy (100%), unless specific queries benefit from the semantic richness of contextual retrieval.
