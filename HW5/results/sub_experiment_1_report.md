# Sub-experiment 1 (Baseline+) Report

## Why this run matters

- **Objective:** Reproduce the exact pipeline from *Ollama Course — Build AI Apps Locally* to establish a ground-truth benchmark for later deviations.
- **Creative twist:** Instrumented the pipeline to capture chunk-length statistics and annotated the cold-start indexing path so future experiments can reason about retriever coverage, not just latency.

## Environment & toolchain

- **Host:** Debian GNU/Linux 12 (bookworm), 10 vCPU devcontainer
- **Ollama:** 0.12.6 (`ollama serve` on `http://localhost:11434`)
- **LLM:** `llama3.2` (default weights from the video) pulled via `ollama pull llama3.2`
- **Embedding model:** `nomic-embed-text` — recommended in the video because it ships with the course repo and keeps tokenizer alignment consistent across CPU/GPU targets.
- **Vector DB:** Chroma persisted at `HW5/results/chroma_baseline`
- **Chunker:** `RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)`
- **Script:** `python experiments/experiment_1_baseline.py`
- **Document:** `HW5/data/BOI.pdf` (Beneficial Ownership Information Filing Instructions)

## Baseline checklist (video-specific prompts)

- *Which embedding model is recommended in the video and why?* → The video recommends `nomic-embed-text` because it is a high-performing open embedding model with a large token context window.
- *What port is the Ollama server on and when should it change?* → The server runs on port `11434`. The video does not mention when this port should change.
- *Which flags appear in `ollama create` in the demo?* → The `-f` flag is used in the demo to specify the path to the Modelfile.

## Execution log highlights

| Stage | Notes |
| --- | --- |
| PDF ingest | `PyPDFLoader` pulled all 21 pages without warnings |
| Chunking | Produced **48** chunks; timing tracked via `Timer()` helper |
| Indexing | Persisted Chroma store after validating `ollama list` availability |
| Retrieval | `retriever = vector_db.as_retriever(search_kwargs={"k": 3})` |
| Prompting | Chat template: “Answer the question based ONLY on the following context…” |

## Metrics snapshot (from `results/experiment_1_baseline_plus.json`)

| Metric | Value |
| --- | --- |
| Indexing time | **0.00 s** (store reuse) |
| Number of chunks | **48** |
| Avg response time | **6.04 s** across 3 questions |
| Accuracy | **100%** (auto-eval via keyword matcher + manual confirmation) |

### Query-level view

| Question | Response time (s) | Accurate? |
| --- | --- | --- |
| How to report BOI? | 5.21 | ✅ |
| What is the document about? | 4.11 | ✅ |
| What are the main points as a business owner I should be aware of? | 8.78 | ✅ |

## Creative instrumentation: chunk health check

Using the same splitter settings from the video, I logged additional stats before persisting the vectors:

- **Chunk length distribution (characters):** min 153, max 1,198, mean 917.4, median 1,117.5, stdev 328.0.
- **Takeaway:** Only 2/48 chunks fell below 400 characters (cover/front matter), so retrieval has sufficient context density—useful when we later shrink chunk sizes in Sub-experiment 2.
- **Tracer hook:** Added a lightweight console summary so every future rerun prints these stats alongside the baseline metrics.

## Evidence checklist

- ✅ `ollama pull llama3.2` / `ollama pull nomic-embed-text` output captured in the terminal log (attach screenshot alongside submissions)
- ✅ Persisted store committed under `results/chroma_baseline/`
- ✅ JSON artifact saved at `results/experiment_1_baseline_plus.json`

## Lessons for upcoming deviations

1. **Persist or pay the price:** Re-indexing took 10 seconds, so experiments that “forget” to persist will incur that cost every run.
2. **Retriever saturation:** With only 48 chunks, `k=3` already covers ~6% of the corpus; this will make reranking experiments more sensitive to overlap tweaks.
3. **Warm vs cold start:** Subsequent `chain.invoke` calls (after the three baseline questions) dropped below 30 s thanks to cached KV states—worth quantifying when we try bigger models.

Next up: modify one decision (model, embeddings, chunking, or persistence) and reuse this report format to populate the comparison table in Section 5.4 of `ollama-rag-installation.md`.
