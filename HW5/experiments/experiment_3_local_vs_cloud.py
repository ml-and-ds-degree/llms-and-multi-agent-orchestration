#!/usr/bin/env -S uv run

"""Sub-experiment 3: Local Ollama vs. Cloud / Hybrid comparison.

This runner executes the RAG pipeline in three modes:
1. local  - All components on the laptop (Ollama LLM + embeddings + Chroma persistence).
2. hybrid - Local embeddings/vector store, but LLM calls pay simulated cloud latency.
3. cloud  - Both embeddings and LLM pay simulated cloud latency and the vector store
             lives in memory (no persistence) to mimic ephemeral SaaS usage.

The "cloud" behaviour uses real Ollama computation underneath (so we can execute
in this offline environment) while injecting deterministic latency and tracking it.
Set the CLOUD_LLM_ENDPOINT/CLOUD_LLM_MODEL environment variables to point to a real
cloud provider if desired; otherwise it falls back to the simulated layer.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama
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

from HW5.queries import evaluate_query_accuracy, get_all_queries, get_baseline_queries
from HW5.utils.metrics import ExperimentMetrics, Timer, print_metrics_summary

BASE_DIR = Path(__file__).resolve().parent.parent


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_DOC_PATH = BASE_DIR / "data" / "BOI.pdf"
DEFAULT_MODEL = "llama3.2"
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
        description="Ollama LLM + embeddings + persisted Chroma",
    ),
    "hybrid": ModeProfile(
        label="Hybrid",
        use_cloud_llm=True,
        use_cloud_embeddings=False,
        persist_store=True,
        description="Local embeddings/vector DB, cloud LLM",
    ),
    "cloud": ModeProfile(
        label="Cloud",
        use_cloud_llm=True,
        use_cloud_embeddings=True,
        persist_store=False,
        description="Embeddings + LLM incur cloud latency, in-memory DB",
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
    simulate_latency_range: Tuple[float, float]


@dataclass
class LatencyTracker:
    label: str
    calls: int = 0
    total_latency: float = 0.0

    def record(self, delay: float) -> None:
        self.calls += 1
        self.total_latency += delay

    def to_dict(self) -> Dict[str, float]:
        avg = self.total_latency / self.calls if self.calls else 0.0
        return {
            "calls": float(self.calls),
            "total_seconds": round(self.total_latency, 3),
            "avg_seconds": round(avg, 3),
        }


class SimulatedCloudEmbeddings:
    def __init__(self, base_embeddings, tracker: LatencyTracker, latency_range):
        self.base = base_embeddings
        self.tracker = tracker
        self.latency_range = latency_range

    def _simulate_latency(self):
        delay = random.uniform(*self.latency_range)
        time.sleep(delay)
        self.tracker.record(delay)

    def embed_documents(self, texts: List[str]):
        self._simulate_latency()
        return self.base.embed_documents(texts)

    def embed_query(self, text: str):
        self._simulate_latency()
        return self.base.embed_query(text)


class SimulatedCloudLLM:
    def __init__(self, base_llm: ChatOllama, tracker: LatencyTracker, latency_range):
        self.base = base_llm
        self.tracker = tracker
        self.latency_range = latency_range

    def _simulate_latency(self):
        delay = random.uniform(*self.latency_range)
        time.sleep(delay)
        self.tracker.record(delay)

    def invoke(self, input, **kwargs):  # noqa: A003
        self._simulate_latency()
        return self.base.invoke(input, **kwargs)

    async def ainvoke(self, input, **kwargs):
        delay = random.uniform(*self.latency_range)
        await asyncio.sleep(delay)
        self.tracker.record(delay)
        return await self.base.ainvoke(input, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base, name)


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
    parser.add_argument(
        "--simulate-latency-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.35, 0.85),
        help="Latency range (seconds) injected when simulating cloud calls",
    )

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
        simulate_latency_range=(
            args.simulate_latency_range[0],
            args.simulate_latency_range[1],
        ),
    )


def run_experiment(config: RunConfig) -> ExperimentMetrics:
    profile = MODE_PROFILES[config.mode_key]
    ensure_ollama_models([config.model_name, config.embedding_model])

    with Timer() as chunk_timer:
        documents = load_documents(config.doc_path)
        chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
    chunk_stats = summarize_chunks(chunks)
    chunk_stats["chunk_time_s"] = chunk_timer.get_elapsed()

    trackers: Dict[str, LatencyTracker] = {}
    base_embeddings = build_embeddings(config.embedding_model)
    embeddings = (
        SimulatedCloudEmbeddings(
            base_embeddings,
            trackers.setdefault(
                "embedding_latency", LatencyTracker(label="cloud_embeddings")
            ),
            config.simulate_latency_range,
        )
        if profile.use_cloud_embeddings
        else base_embeddings
    )
    persist_target = config.persist_dir if profile.persist_store else None
    vector_db, indexing_time, store_mode = build_vector_store(
        chunks,
        embeddings,
        config.vector_store_name,
        persist_target,
        config.force_reindex,
    )
    base_llm = build_llm(config.model_name)
    llm = (
        SimulatedCloudLLM(
            base_llm,
            trackers.setdefault("llm_latency", LatencyTracker(label="cloud_llm")),
            config.simulate_latency_range,
        )
        if profile.use_cloud_llm
        else base_llm
    )
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
            "cloud_latency": {
                k: {"label": tracker.label, **tracker.to_dict()}
                for k, tracker in trackers.items()
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
    logger.info("\n✅ Sub-experiment 3 run completed (%s mode)!", config.mode_key)
    logger.info("Results saved to: %s", config.output_path)


if __name__ == "__main__":
    main()
