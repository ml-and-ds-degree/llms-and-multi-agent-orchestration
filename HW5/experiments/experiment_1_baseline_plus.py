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
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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


def ensure_ollama_up(config: RunConfig) -> None:
    logger.info("Checking Ollama server on localhost:11434")
    try:
        ollama.list()
    except Exception as exc:  # pragma: no cover - network validation
        raise RuntimeError(
            "Ollama server not reachable. Run 'ollama serve' locally first."
        ) from exc

    for model in {config.model_name, config.embedding_model}:
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
        "stdev": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
        "p10": percentile(sorted_lengths, 0.10),
        "p25": percentile(sorted_lengths, 0.25),
        "p75": percentile(sorted_lengths, 0.75),
        "p90": percentile(sorted_lengths, 0.90),
        "sample_first5": lengths[:5],
    }
    return summary


def build_vector_store(chunks, config: RunConfig) -> Tuple[Chroma, float, str]:
    embeddings = OllamaEmbeddings(model=config.embedding_model)

    if config.no_persist or config.persist_dir is None:
        logger.info("Building in-memory Chroma store (no persistence)")
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


def create_chain(vector_db: Chroma, config: RunConfig):
    llm = ChatOllama(model=config.model_name)
    retriever = vector_db.as_retriever(search_kwargs={"k": config.retriever_k})

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


def run_experiment(config: RunConfig) -> ExperimentMetrics:
    ensure_ollama_up(config)

    with Timer() as chunk_timer:
        _, chunks = ingest_and_split(config)
    chunk_stats = summarize_chunks(chunks)
    chunk_stats["chunk_time_s"] = chunk_timer.get_elapsed()

    vector_db, indexing_time, store_mode = build_vector_store(chunks, config)

    chain = create_chain(vector_db, config)
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

    queries_metrics, latency_stats = run_queries(chain, query_set)
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
