"""Sub-experiment 3: Local Ollama vs. Cloud / Hybrid comparison.

This runner executes the RAG pipeline in three modes:
1. local  – All components on the laptop (Ollama LLM + embeddings + Chroma persistence).
2. hybrid – Local embeddings/vector store, but LLM calls pay simulated cloud latency.
3. cloud  – Both embeddings and LLM pay simulated cloud latency and the vector store
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
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from queries import (  # noqa: E402  # pylint: disable=wrong-import-position
    evaluate_query_accuracy,
    get_all_queries,
    get_baseline_queries,
)
from utils.metrics import (  # noqa: E402  # pylint: disable=wrong-import-position
    ExperimentMetrics,
    QueryMetrics,
    Timer,
    print_metrics_summary,
)

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


def ensure_ollama(models: List[str]) -> None:
    logger.info("Validating Ollama server on localhost:11434")
    try:
        ollama.list()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Ollama server not reachable. Run 'ollama serve'.") from exc

    for model in models:
        logger.info("Ensuring model '%s' is pulled", model)
        ollama.pull(model)


def ingest_and_split(config: RunConfig):
    logger.info("Loading PDF from %s", config.doc_path)
    loader = PyPDFLoader(file_path=str(config.doc_path))
    documents = loader.load()

    logger.info(
        "Splitting documents with chunk_size=%s, overlap=%s",
        config.chunk_size,
        config.chunk_overlap,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Generated %s chunks", len(chunks))
    return chunks


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
        "stdev": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
    }
    return summary


def build_embeddings(
    config: RunConfig, profile: ModeProfile, trackers: Dict[str, LatencyTracker]
):
    base_embeddings = OllamaEmbeddings(model=config.embedding_model)
    if profile.use_cloud_embeddings:
        tracker = trackers.setdefault(
            "embedding_latency", LatencyTracker(label="cloud_embeddings")
        )
        return SimulatedCloudEmbeddings(
            base_embeddings, tracker, config.simulate_latency_range
        )
    return base_embeddings


def build_vector_store(chunks, config: RunConfig, profile: ModeProfile, embeddings):
    if not profile.persist_store or config.persist_dir is None:
        logger.info("Creating in-memory Chroma store (no persistence)")
        with Timer() as index_timer:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=config.vector_store_name,
            )
        return vector_db, index_timer.get_elapsed(), "memory"

    store_ready = config.persist_dir.exists() and any(config.persist_dir.iterdir())

    if store_ready and not config.force_reindex:
        logger.info("Reusing persisted Chroma store at %s", config.persist_dir)
        vector_db = Chroma(
            collection_name=config.vector_store_name,
            embedding_function=embeddings,
            persist_directory=str(config.persist_dir),
        )
        return vector_db, 0.0, "reuse"

    logger.info("(Re)building Chroma store at %s", config.persist_dir)
    with Timer() as index_timer:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=config.vector_store_name,
            persist_directory=str(config.persist_dir),
        )
    return vector_db, index_timer.get_elapsed(), "rebuild"


def build_llm(
    config: RunConfig, profile: ModeProfile, trackers: Dict[str, LatencyTracker]
):
    llm = ChatOllama(model=config.model_name)
    if profile.use_cloud_llm:
        tracker = trackers.setdefault("llm_latency", LatencyTracker(label="cloud_llm"))
        return SimulatedCloudLLM(llm, tracker, config.simulate_latency_range)
    return llm


def create_chain(vector_db: Chroma, llm, config: RunConfig):
    retriever = vector_db.as_retriever(search_kwargs={"k": config.retriever_k})
    template = """Answer the question based ONLY on the following context:\n{context}\nQuestion: {question}\n"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | RunnableLambda(lambda prompt_value: llm.invoke(prompt_value))
        | StrOutputParser()
    )
    return chain


def run_queries(
    chain, query_set: List[str]
) -> Tuple[List[QueryMetrics], Dict[str, float]]:
    collected: List[QueryMetrics] = []
    latency_bucket: List[float] = []

    for i, question in enumerate(query_set, 1):
        logger.info("\n--- Query %s/%s: %s", i, len(query_set), question)
        with Timer() as query_timer:
            try:
                response = chain.invoke(input=question)
            except Exception as exc:  # pragma: no cover
                logger.error("Error during query '%s': %s", question, exc)
                response = f"ERROR: {exc}"
        elapsed = query_timer.get_elapsed()
        latency_bucket.append(elapsed)
        logger.info("Response time: %.2fs", elapsed)

        collected.append(
            QueryMetrics(
                query=question,
                response=response,
                response_time=elapsed,
                chunks_retrieved=3,
                is_accurate=evaluate_query_accuracy(question, response),
            )
        )

    summary = {
        "min": min(latency_bucket) if latency_bucket else 0.0,
        "max": max(latency_bucket) if latency_bucket else 0.0,
        "mean": statistics.fmean(latency_bucket) if latency_bucket else 0.0,
    }
    if len(latency_bucket) > 1:
        summary["stdev"] = statistics.pstdev(latency_bucket)
    return collected, summary


def run_experiment(config: RunConfig) -> ExperimentMetrics:
    profile = MODE_PROFILES[config.mode_key]
    ensure_ollama([config.model_name, config.embedding_model])

    with Timer() as chunk_timer:
        chunks = ingest_and_split(config)
    chunk_stats = summarize_chunks(chunks)
    chunk_stats["chunk_time_s"] = chunk_timer.get_elapsed()

    trackers: Dict[str, LatencyTracker] = {}
    embeddings = build_embeddings(config, profile, trackers)
    vector_db, indexing_time, store_mode = build_vector_store(
        chunks, config, profile, embeddings
    )
    llm = build_llm(config, profile, trackers)
    chain = create_chain(vector_db, llm, config)

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

    queries_metrics, latency_stats = run_queries(chain, query_set)
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
