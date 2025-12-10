#!/usr/bin/env -S uv run

"""Sub-experiment 3: Local Ollama vs. Cloud / Hybrid comparison.

This runner executes the RAG pipeline in two modes:
1. local  - All components on the laptop (Ollama LLM + embeddings + Chroma persistence).
2. hybrid - Local embeddings/vector store, but LLM calls go to Ollama Cloud (gpt-oss:120b-cloud).

The "hybrid" mode uses real Ollama Cloud API.
You MUST set OLLAMA_API_KEY environment variable for hybrid mode.

Note: Cloud embeddings are not currently available on Ollama Cloud, so we only support hybrid mode.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from utils.rag_pipeline import (
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

from queries import evaluate_query_accuracy, get_all_queries, get_baseline_queries
from utils.metrics import ExperimentMetrics, Timer, print_metrics_summary

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_DOC_PATH = BASE_DIR / "data" / "BOI.pdf"
DEFAULT_MODEL = "llama3.2"
CLOUD_MODEL = "gpt-oss:120b-cloud"
DEFAULT_EMBEDDING = "nomic-embed-text"
DEFAULT_PERSIST_DIR = BASE_DIR / "results" / "chroma_baseline"
DEFAULT_VECTOR_STORE = "rag-cloud"
DEFAULT_OUTPUT = BASE_DIR / "results" / "experiment_3_local.json"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_RETRIEVER_K = 3


@dataclass
class ModeProfile:
    label: str
    use_cloud_llm: bool
    use_cloud_embeddings: bool
    persist_store: bool
    description: str


MODE_PROFILES: Dict[str, ModeProfile] = {
    "local": ModeProfile(
        label="All Local",
        use_cloud_llm=False,
        use_cloud_embeddings=False,
        persist_store=True,
        description="Local Ollama LLM (llama3.2) + embeddings + persisted Chroma",
    ),
    "hybrid": ModeProfile(
        label="Hybrid (Cloud)",
        use_cloud_llm=True,
        use_cloud_embeddings=False,
        persist_store=True,
        description="Local embeddings/vector DB, Cloud LLM (gpt-oss:120b-cloud)",
    ),
}


@dataclass
class RunConfig:
    mode_key: str
    doc_path: Path
    model_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    persist_dir: Optional[Path]
    vector_store_name: str
    output_path: Path
    force_reindex: bool
    use_all_queries: bool
    ollama_api_key: Optional[str]


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Compare local vs cloud RAG modes")
    parser.add_argument("--mode", choices=list(MODE_PROFILES.keys()), default="local")
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--retriever-k", type=int, default=DEFAULT_RETRIEVER_K)
    parser.add_argument("--vector-store-name", default=DEFAULT_VECTOR_STORE)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--all-queries", action="store_true")

    args = parser.parse_args()

    doc_path = (
        args.doc_path if args.doc_path.is_absolute() else BASE_DIR / args.doc_path
    )
    persist_dir = (
        None
        if not MODE_PROFILES[args.mode].persist_store
        else args.persist_dir
        if args.persist_dir.is_absolute()
        else BASE_DIR / args.persist_dir
    )
    output_path = args.output if args.output.is_absolute() else BASE_DIR / args.output

    return RunConfig(
        mode_key=args.mode,
        doc_path=doc_path,
        model_name=args.model_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retriever_k=args.retriever_k,
        persist_dir=persist_dir,
        vector_store_name=args.vector_store_name,
        output_path=output_path,
        force_reindex=args.force_reindex,
        use_all_queries=args.all_queries,
        ollama_api_key=os.environ.get("OLLAMA_API_KEY"),
    )


def run_experiment(config: RunConfig) -> ExperimentMetrics:
    profile = MODE_PROFILES[config.mode_key]

    # Only pull models if we are running locally (or hybrid local component)
    if not profile.use_cloud_llm or not profile.use_cloud_embeddings:
        ensure_ollama_models([config.model_name, config.embedding_model])

    with Timer() as chunk_timer:
        documents = load_documents(config.doc_path)
        chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
    chunk_stats = summarize_chunks(chunks)
    chunk_stats["chunk_time_s"] = chunk_timer.get_elapsed()

    # Configure Cloud settings
    cloud_base_url = "https://ollama.com" if config.ollama_api_key else None
    cloud_headers = (
        {"Authorization": f"Bearer {config.ollama_api_key}"}
        if config.ollama_api_key
        else None
    )

    if profile.use_cloud_embeddings:
        if not config.ollama_api_key:
            raise ValueError("OLLAMA_API_KEY env var is required for cloud embeddings")

        logger.info("Using REAL Cloud Embeddings (Ollama Cloud)")
        embeddings = build_embeddings(
            config.embedding_model, base_url=cloud_base_url, headers=cloud_headers
        )
    else:
        embeddings = build_embeddings(config.embedding_model)

    persist_target = config.persist_dir if profile.persist_store else None
    vector_db, indexing_time, store_mode = build_vector_store(
        chunks,
        embeddings,
        config.vector_store_name,
        persist_target,
        config.force_reindex,
    )

    if profile.use_cloud_llm:
        if not config.ollama_api_key:
            raise ValueError("OLLAMA_API_KEY env var is required for cloud LLM")

        # Use the specific cloud model constant for cloud/hybrid modes
        target_model = CLOUD_MODEL
        logger.info("Using REAL Cloud LLM (Ollama Cloud) - Model: %s", target_model)
        llm = build_llm(target_model, base_url=cloud_base_url, headers=cloud_headers)
    else:
        llm = build_llm(config.model_name)

    retriever = build_retriever(vector_db, retriever_k=config.retriever_k)
    chain = build_chain(
        retriever,
        llm,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        llm_invoker=getattr(llm, "invoke"),
    )

    query_set = get_all_queries() if config.use_all_queries else get_baseline_queries()

    metrics = ExperimentMetrics(
        experiment_name=f"Experiment 3: {profile.label}",
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
            "mode": profile.label,
            "description": profile.description,
            "vector_store_mode": store_mode,
            "chunk_stats": chunk_stats,
            "query_set": "extended" if config.use_all_queries else "baseline",
            "latency_snapshot": latency_stats,
        }
    )

    print_metrics_summary(metrics)
    metrics.save(str(config.output_path))
    return metrics


def main():
    config = parse_args()
    os.chdir(BASE_DIR)
    metrics = run_experiment(config)
    logger.info("\n✅ Sub-experiment 3 run completed (%s mode)!", config.mode_key)
    logger.info("Results saved to: %s", config.output_path)


if __name__ == "__main__":
    main()
