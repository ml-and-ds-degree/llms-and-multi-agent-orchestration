"""Sub-experiment 1 "Baseline+" run with instrumentation.

This script mirrors the video-aligned baseline pipeline while adding:
- Optional reuse of a persisted Chroma store (warm start) unless --force-reindex is set.
- Chunk-length statistics to reason about coverage before/after tweaks.
- CLI arguments for extended queries and custom artifact output paths.

Usage examples:
    python experiments/experiment_1_baseline_plus.py
    python experiments/experiment_1_baseline_plus.py --force-reindex --all-queries
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add HW5 root to PYTHONPATH for absolute-style imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from queries import (
    evaluate_query_accuracy,
    get_all_queries,
    get_baseline_queries,
)
from utils.metrics import (
    ExperimentMetrics,
    QueryMetrics,
    Timer,
    print_metrics_summary,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DOC_PATH = BASE_DIR / "data" / "BOI.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIR = BASE_DIR / "results" / "chroma_baseline"
PORT = 11434
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
DEFAULT_OUTPUT = BASE_DIR / "results" / "experiment_1_baseline_plus.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instrumented Baseline RAG run")
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Ignore existing Chroma store and rebuild from source document",
    )
    parser.add_argument(
        "--all-queries",
        action="store_true",
        help="Run the extended query set instead of just the three baseline prompts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the metrics JSON output",
    )
    parser.add_argument(
        "--doc-path",
        type=Path,
        default=DOC_PATH,
        help="Path to the PDF document to ingest",
    )
    return parser.parse_args()


def ensure_ollama_up() -> None:
    logger.info("Checking Ollama server on localhost:%s", PORT)
    try:
        ollama.list()
    except Exception as exc:  # pragma: no cover - network validation
        raise RuntimeError(
            "Ollama server not reachable. Run 'ollama serve' locally first."
        ) from exc

    for model in {MODEL_NAME, EMBEDDING_MODEL}:
        logger.info("Ensuring model '%s' is pulled", model)
        ollama.pull(model)


def ingest_and_split(doc_path: Path):
    logger.info("Loading PDF from %s", doc_path)
    loader = PyPDFLoader(file_path=str(doc_path))
    documents = loader.load()

    logger.info(
        "Splitting documents with chunk_size=%s, overlap=%s",
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Generated %s chunks", len(chunks))
    return documents, chunks


def percentile(values: List[int], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    k = (len(values) - 1) * fraction
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values[int(k)])
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return float(d0 + d1)


def summarize_chunks(chunks) -> Dict[str, float]:  # type: ignore[no-untyped-def]
    lengths = [len(chunk.page_content) for chunk in chunks]
    if not lengths:
        return {"count": 0}

    sorted_lengths = sorted(lengths)
    summary = {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "stdev": statistics.pstdev(lengths),
        "p10": percentile(sorted_lengths, 0.10),
        "p25": percentile(sorted_lengths, 0.25),
        "p75": percentile(sorted_lengths, 0.75),
        "p90": percentile(sorted_lengths, 0.90),
        "sample_first5": lengths[:5],
    }
    return summary


def build_vector_store(
    chunks,
    force_reindex: bool,
) -> Tuple[Chroma, float, bool]:
    store_ready = PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if store_ready and not force_reindex:
        logger.info("Reusing persisted Chroma store at %s", PERSIST_DIR)
        vector_db = Chroma(
            collection_name=VECTOR_STORE_NAME,
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )
        return vector_db, 0.0, True

    logger.info("(Re)building Chroma store at %s", PERSIST_DIR)
    with Timer() as index_timer:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=str(PERSIST_DIR),
        )
    return vector_db, index_timer.get_elapsed(), False


def create_chain(vector_db: Chroma):
    llm = ChatOllama(model=MODEL_NAME)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    template = """Answer the question based ONLY on the following context:\n{context}\nQuestion: {question}\n"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def run_queries(
    chain, queries: List[str]
) -> Tuple[List[QueryMetrics], Dict[str, float]]:
    collected: List[QueryMetrics] = []
    latency_bucket: List[float] = []

    for i, question in enumerate(queries, 1):
        logger.info("\n--- Query %s/%s: %s", i, len(queries), question)
        with Timer() as query_timer:
            try:
                response = chain.invoke(input=question)
            except Exception as exc:  # pragma: no cover - runtime errors
                logger.error("Error during query '%s': %s", question, exc)
                response = f"ERROR: {exc}"
        elapsed = query_timer.get_elapsed()
        latency_bucket.append(elapsed)
        logger.info("Response time: %.2fs", elapsed)

        query_metrics = QueryMetrics(
            query=question,
            response=response,
            response_time=elapsed,
            chunks_retrieved=3,
            is_accurate=evaluate_query_accuracy(question, response),
        )
        collected.append(query_metrics)

    latency_summary = {
        "min": min(latency_bucket) if latency_bucket else 0.0,
        "max": max(latency_bucket) if latency_bucket else 0.0,
        "mean": statistics.fmean(latency_bucket) if latency_bucket else 0.0,
    }
    if len(latency_bucket) >= 2:
        latency_summary["stdev"] = statistics.pstdev(latency_bucket)
    return collected, latency_summary


def run_experiment(args: argparse.Namespace) -> ExperimentMetrics:
    ensure_ollama_up()

    with Timer() as chunk_timer:
        _, chunks = ingest_and_split(args.doc_path)
    chunk_stats = summarize_chunks(chunks)
    chunk_stats["chunk_time_s"] = chunk_timer.get_elapsed()

    vector_db, indexing_time, reused_store = build_vector_store(
        chunks=chunks,
        force_reindex=args.force_reindex,
    )

    chain = create_chain(vector_db)
    query_set = get_all_queries() if args.all_queries else get_baseline_queries()

    metrics = ExperimentMetrics(
        experiment_name="Experiment 1: Baseline+",
        model_name=MODEL_NAME,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        persist_directory=str(PERSIST_DIR.resolve()),
    )
    metrics.num_chunks = int(chunk_stats.get("count", 0))
    metrics.indexing_time = indexing_time

    queries_metrics, latency_stats = run_queries(chain, query_set)
    metrics.queries.extend(queries_metrics)
    metrics.metadata.update(
        {
            "vector_store_mode": "reuse" if reused_store else "rebuild",
            "chunk_stats": chunk_stats,
            "query_set": "extended" if args.all_queries else "baseline",
            "latency_snapshot": latency_stats,
        }
    )

    print_metrics_summary(metrics)
    metrics.save(str(args.output))
    return metrics


def main():
    args = parse_args()
    os.chdir(BASE_DIR)
    metrics = run_experiment(args)
    logger.info("\n✅ Instrumented baseline experiment completed!")
    logger.info("Results saved to: %s", args.output)
    logger.info("Vector store directory: %s", PERSIST_DIR)


if __name__ == "__main__":
    main()
