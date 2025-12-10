#!/usr/bin/env -S uv run

"""Sub-experiment 4: Contextual retrieval + reranking techniques."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.documents import Document
from utils.rerankers import llm_rerank

from HW5.queries import evaluate_query_accuracy, get_all_queries, get_baseline_queries
from HW5.utils.contextualizer import build_contextual_chunks
from HW5.utils.metrics import ExperimentMetrics, Timer, print_metrics_summary
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DOC_PATH = BASE_DIR / "data" / "BOI.pdf"
DEFAULT_MODEL = "llama3.2"
DEFAULT_EMBEDDING = "nomic-embed-text"
DEFAULT_PERSIST_DIR = BASE_DIR / "results" / "chroma_advanced"
DEFAULT_VECTOR_STORE = "rag-advanced"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_RETRIEVER_K = 3
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"


@dataclass
class RunConfig:
    variant: str
    doc_path: Path
    model_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    vector_store_name: str
    persist_dir: Optional[Path]
    retriever_k: int
    extended_queries: bool
    force_reindex: bool
    output_path: Path
    contextual_summary_model: Optional[str]
    rerank_top_k: int
    rerank_candidates: int


VARIANTS = {"basic", "contextual", "rerank"}


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Experiment 4 advanced RAG")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="basic",
        help="Which configuration to run",
    )
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--vector-store-name", default=DEFAULT_VECTOR_STORE)
    parser.add_argument("--retriever-k", type=int, default=DEFAULT_RETRIEVER_K)
    parser.add_argument("--extended-queries", action="store_true")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "experiment_4_basic.json",
    )
    parser.add_argument(
        "--contextual-summary-model",
        default=None,
        help="Optional alternate model to summarize chunks",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=3,
        help="Top-k documents to keep after reranking",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=10,
        help="How many candidates to retrieve before reranking",
    )

    args = parser.parse_args()

    doc_path = (
        args.doc_path if args.doc_path.is_absolute() else BASE_DIR / args.doc_path
    )
    persist_dir = (
        args.persist_dir
        if args.persist_dir.is_absolute()
        else BASE_DIR / args.persist_dir
    )

    # If the user didn't override output, namespace by variant for clarity.
    if args.output == DEFAULT_OUTPUT_DIR / "experiment_4_basic.json":
        suggested_name = DEFAULT_OUTPUT_DIR / f"experiment_4_{args.variant}.json"
        output_path = suggested_name
    else:
        output_path = (
            args.output if args.output.is_absolute() else BASE_DIR / args.output
        )

    return RunConfig(
        variant=args.variant,
        doc_path=doc_path,
        model_name=args.model_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_name=args.vector_store_name,
        persist_dir=persist_dir,
        retriever_k=args.retriever_k,
        extended_queries=args.extended_queries,
        force_reindex=args.force_reindex,
        output_path=output_path,
        contextual_summary_model=args.contextual_summary_model,
        rerank_top_k=args.rerank_top_k,
        rerank_candidates=args.rerank_candidates,
    )


def _maybe_contextualize_chunks(
    *,
    variant: str,
    chunks: List[Document],
    model_name: str,
    contextual_summary_model: Optional[str],
) -> List[Document]:
    if variant != "contextual":
        return chunks

    summary_model = contextual_summary_model or model_name
    helper_llm = build_llm(summary_model)

    def call_llm(prompt: str) -> str:
        result = helper_llm.invoke(prompt)
        return str(getattr(result, "content", result))

    return build_contextual_chunks(chunks, summary_fn=call_llm)


def _maybe_configure_reranker(
    *,
    variant: str,
    llm,
    config: RunConfig,
) -> Dict[str, Callable]:
    if variant != "rerank":
        return {}

    candidates_k = max(config.retriever_k, config.rerank_candidates)

    def rerank_fn(documents: List[Document], question: str) -> List[Document]:
        def scorer(prompt: str) -> str:
            result = llm.invoke(prompt)
            return str(getattr(result, "content", result))

        return llm_rerank(
            documents,
            question,
            scorer,
            top_k=config.rerank_top_k,
        )

    return {
        "reranker": rerank_fn,
        "retriever_k": candidates_k,
    }


def run_experiment(config: RunConfig) -> ExperimentMetrics:
    ensure_ollama_models([config.model_name, config.embedding_model])
    if (
        config.contextual_summary_model
        and config.contextual_summary_model != config.model_name
    ):
        ensure_ollama_models([config.contextual_summary_model])

    with Timer() as chunk_timer:
        documents = load_documents(config.doc_path)
        chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
    chunk_stats = summarize_chunks(chunks)
    chunk_stats["chunk_time_s"] = chunk_timer.get_elapsed()

    chunks = _maybe_contextualize_chunks(
        variant=config.variant,
        chunks=chunks,
        model_name=config.model_name,
        contextual_summary_model=config.contextual_summary_model,
    )

    embeddings = build_embeddings(config.embedding_model)
    vector_db, indexing_time, store_mode = build_vector_store(
        chunks,
        embeddings,
        config.vector_store_name,
        config.persist_dir,
        config.force_reindex,
    )

    llm = build_llm(config.model_name)
    rerank_kwargs = _maybe_configure_reranker(
        variant=config.variant,
        llm=llm,
        config=config,
    )
    effective_k = rerank_kwargs.get("retriever_k", config.retriever_k)
    retriever = build_retriever(vector_db, retriever_k=effective_k)
    chain = build_chain(
        retriever,
        llm,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        reranker=rerank_kwargs.get("reranker"),
    )

    query_set = get_all_queries() if config.extended_queries else get_baseline_queries()

    metrics = ExperimentMetrics(
        experiment_name=f"Experiment 4: {config.variant}",
        model_name=config.model_name,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        persist_directory=str(config.persist_dir),
    )
    metrics.num_chunks = int(chunk_stats.get("count", 0))
    metrics.indexing_time = indexing_time

    chunks_reported = (
        config.rerank_top_k if config.variant == "rerank" else config.retriever_k
    )
    queries_metrics, latency_stats = run_query_batch(
        chain,
        query_set,
        evaluate_query_accuracy,
        chunks_retrieved=chunks_reported,
        logger_obj=logger,
    )
    metrics.queries.extend(queries_metrics)
    metrics.metadata.update(
        {
            "variant": config.variant,
            "vector_store_mode": store_mode,
            "chunk_stats": chunk_stats,
            "query_set": "extended" if config.extended_queries else "baseline",
            "latency_snapshot": latency_stats,
            "contextual_summary_model": config.contextual_summary_model,
            "rerank_top_k": config.rerank_top_k,
            "rerank_candidates": config.rerank_candidates,
        }
    )

    print_metrics_summary(metrics)
    metrics.save(str(config.output_path))
    return metrics


def main():
    config = parse_args()
    os.chdir(BASE_DIR)
    run_experiment(config)


if __name__ == "__main__":
    main()
