"""Instrumented baseline experiment runner with configurable variations.

This script defaults to the exact configuration from the
"Ollama Course — Build AI Apps Locally" baseline but exposes CLI
flags so we can drive Sub-experiment 2 (and beyond) without duplicating
pipeline code.

Example invocations:
    # Baseline (same as video)
    python experiments/experiment_1_baseline_plus.py

    # Change the LLM only (Sub-experiment 2 option 1)
    python experiments/experiment_1_baseline_plus.py \
        --model-name llama3.2:3b --persist-dir results/chroma_llama3_3b \
        --output results/experiment_2_llm_small.json --force-reindex

    # Change chunking + skip persistence (Sub-experiment 2 option 2)
    python experiments/experiment_1_baseline_plus.py \
        --chunk-size 300 --chunk-overlap 50 --no-persist --force-reindex \
        --output results/experiment_2_chunky.json
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from HW5.utils.rag_pipeline import (
    DEFAULT_PROMPT_TEMPLATE,
    build_chain,
    build_embeddings,
    build_llm,
    build_retriever,
    build_vector_store,
    ensure_ollama_models,
    load_documents,
    run_query_batch,
    split_documents,
    summarize_chunks,
)

BASE_DIR = Path(__file__).resolve().parent.parent


from queries import (  # noqa: E402  # pylint: disable=wrong-import-position
    evaluate_query_accuracy,
    get_all_queries,
    get_baseline_queries,
)
from utils.metrics import (  # noqa: E402  # pylint: disable=wrong-import-position
    ExperimentMetrics,
    Timer,
    print_metrics_summary,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_DOC_PATH = BASE_DIR / "data" / "BOI.pdf"
DEFAULT_MODEL = "llama3.2"
DEFAULT_EMBEDDING = "nomic-embed-text"
DEFAULT_VECTOR_STORE = "simple-rag"
DEFAULT_PERSIST_DIR = BASE_DIR / "results" / "chroma_baseline"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_OUTPUT = BASE_DIR / "results" / "experiment_1_baseline_plus.json"
DEFAULT_RETRIEVER_K = 3


@dataclass
class RunConfig:
    doc_path: Path
    model_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    vector_store_name: str
    persist_dir: Optional[Path]
    retriever_k: int
    force_reindex: bool
    use_all_queries: bool
    output_path: Path
    no_persist: bool


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Instrumented Baseline RAG run")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--vector-store-name", default=DEFAULT_VECTOR_STORE)
    parser.add_argument("--retriever-k", type=int, default=DEFAULT_RETRIEVER_K)
    parser.add_argument("--no-persist", action="store_true")

    args = parser.parse_args()

    doc_path = (
        args.doc_path if args.doc_path.is_absolute() else BASE_DIR / args.doc_path
    )
    persist_dir: Optional[Path]
    if args.no_persist:
        persist_dir = None
    else:
        persist_dir = (
            args.persist_dir
            if args.persist_dir.is_absolute()
            else BASE_DIR / args.persist_dir
        )
    output_path = args.output if args.output.is_absolute() else BASE_DIR / args.output

    return RunConfig(
        doc_path=doc_path,
        model_name=args.model_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_name=args.vector_store_name,
        persist_dir=persist_dir,
        retriever_k=args.retriever_k,
        force_reindex=args.force_reindex,
        use_all_queries=args.all_queries,
        output_path=output_path,
        no_persist=args.no_persist,
    )


def run_experiment(config: RunConfig) -> ExperimentMetrics:
    ensure_ollama_models([config.model_name, config.embedding_model])

    with Timer() as chunk_timer:
        documents = load_documents(config.doc_path)
        chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
    chunk_stats = summarize_chunks(chunks)
    chunk_stats["chunk_time_s"] = chunk_timer.get_elapsed()

    embeddings = build_embeddings(config.embedding_model)
    persist_target = None if config.no_persist else config.persist_dir
    vector_db, indexing_time, store_mode = build_vector_store(
        chunks,
        embeddings,
        config.vector_store_name,
        persist_target,
        config.force_reindex,
    )

    llm = build_llm(config.model_name)
    retriever = build_retriever(vector_db, retriever_k=config.retriever_k)
    chain = build_chain(retriever, llm, prompt_template=DEFAULT_PROMPT_TEMPLATE)
    query_set = get_all_queries() if config.use_all_queries else get_baseline_queries()

    experiment_label = (
        "Experiment 2: Variation"
        if (
            config.model_name != DEFAULT_MODEL
            or config.embedding_model != DEFAULT_EMBEDDING
            or config.chunk_size != DEFAULT_CHUNK_SIZE
            or config.chunk_overlap != DEFAULT_CHUNK_OVERLAP
            or config.retriever_k != DEFAULT_RETRIEVER_K
            or config.no_persist
        )
        else "Experiment 1: Baseline+"
    )

    metrics = ExperimentMetrics(
        experiment_name=experiment_label,
        model_name=config.model_name,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        persist_directory=str(config.persist_dir.resolve())
        if config.persist_dir
        else None,
    )
    metrics.num_chunks = int(chunk_stats.get("count", 0))
    metrics.indexing_time = indexing_time

    queries_metrics, latency_stats = run_query_batch(
        chain,
        query_set,
        evaluate_query_accuracy,
        chunks_retrieved=config.retriever_k,
        logger_obj=logger,
    )
    metrics.queries.extend(queries_metrics)
    metrics.metadata.update(
        {
            "vector_store_mode": store_mode,
            "chunk_stats": chunk_stats,
            "query_set": "extended" if config.use_all_queries else "baseline",
            "latency_snapshot": latency_stats,
            "config": {
                "model": config.model_name,
                "embedding": config.embedding_model,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "retriever_k": config.retriever_k,
                "persist_dir": str(config.persist_dir) if config.persist_dir else None,
                "no_persist": config.no_persist,
            },
        }
    )

    print_metrics_summary(metrics)
    metrics.save(str(config.output_path))
    return metrics


def main():
    config = parse_args()
    os.chdir(BASE_DIR)
    metrics = run_experiment(config)
    logger.info("\n✅ Instrumented experiment completed!")
    logger.info("Results saved to: %s", config.output_path)
    if config.persist_dir:
        logger.info("Vector store directory: %s", config.persist_dir)


if __name__ == "__main__":
    main()
