"""
Shared RAG pipeline utilities extracted from the experiment runners.
Provides reusable building blocks for loading documents, chunking,
vector store management, chain construction, and query execution.
"""

from __future__ import annotations

import logging
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .metrics import QueryMetrics, Timer

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = (
    "Answer the question based ONLY on the following context:\n"
    "{context}\n"
    "Question: {question}\n"
)


def ensure_ollama_models(models: Iterable[str]) -> None:
    """Validate Ollama availability and pull required models."""
    logger.info("Checking Ollama server on localhost:11434")
    try:
        ollama.list()
    except Exception as exc:  # pragma: no cover - external service
        raise RuntimeError(
            "Ollama server not reachable. Run `ollama serve` locally first."
        ) from exc

    for model in models:
        logger.info("Ensuring model '%s' is pulled", model)
        ollama.pull(model)


def load_documents(doc_path: Path):
    """Load PDF documents from the given path."""
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    logger.info("Loading PDF from %s", doc_path)
    loader = PyPDFLoader(file_path=str(doc_path))
    return loader.load()


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    """Split documents into text chunks."""
    logger.info(
        "Splitting documents with chunk_size=%s, overlap=%s",
        chunk_size,
        chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Generated %s chunks", len(chunks))
    return chunks


def _percentile(values: List[int], fraction: float) -> float:
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
    """Return descriptive stats for chunk lengths."""
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
        "p10": _percentile(sorted_lengths, 0.10),
        "p25": _percentile(sorted_lengths, 0.25),
        "p75": _percentile(sorted_lengths, 0.75),
        "p90": _percentile(sorted_lengths, 0.90),
        "sample_first5": lengths[:5],
    }
    return summary


def build_embeddings(model_name: str):
    """Create an Ollama embeddings runnable."""
    return OllamaEmbeddings(model=model_name)


def build_vector_store(
    chunks,
    embeddings,
    vector_store_name: str,
    persist_dir: Optional[Path],
    force_reindex: bool = False,
) -> Tuple[Chroma, float, str]:
    """
    Create or reuse a Chroma vector store.

    Returns (vector_db, indexing_time_seconds, mode) where mode is one of
    "rebuild", "reuse", or "memory".
    """
    if persist_dir is None:
        logger.info("Building in-memory Chroma store (no persistence)")
        with Timer() as index_timer:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=vector_store_name,
            )
        return vector_db, index_timer.get_elapsed(), "memory"

    persist_dir.mkdir(parents=True, exist_ok=True)
    store_ready = persist_dir.exists() and any(persist_dir.iterdir())

    if store_ready and not force_reindex:
        logger.info("Reusing persisted Chroma store at %s", persist_dir)
        vector_db = Chroma(
            collection_name=vector_store_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        return vector_db, 0.0, "reuse"

    logger.info("(Re)building Chroma store at %s", persist_dir)
    with Timer() as index_timer:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=vector_store_name,
            persist_directory=str(persist_dir),
        )
    return vector_db, index_timer.get_elapsed(), "rebuild"


def build_llm(model_name: str) -> ChatOllama:
    """Instantiate the chat model."""
    return ChatOllama(model=model_name)


def build_retriever(vector_db: Chroma, retriever_k: int):
    """Create a retriever for the given vector store."""
    return vector_db.as_retriever(search_kwargs={"k": retriever_k})


def build_chain(
    retriever,
    llm,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    llm_invoker: Optional[Callable[[Any], Any]] = None,
):
    """
    Assemble the RAG chain. If llm_invoker is provided, it is wrapped in a
    RunnableLambda so non-LangChain-native LLM adapters can be used.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_stage = (
        RunnableLambda(lambda prompt_value: llm_invoker(prompt_value))
        if llm_invoker
        else llm
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_stage
        | StrOutputParser()
    )
    return chain


def run_query_batch(
    chain: Runnable,
    queries: List[str],
    accuracy_fn: Optional[Callable[[str, str], Optional[bool]]] = None,
    *,
    chunks_retrieved: int,
    logger_obj: Optional[logging.Logger] = None,
) -> Tuple[List[QueryMetrics], Dict[str, float]]:
    """Run a batch of queries through a chain and collect metrics."""
    log = logger_obj or logger
    collected: List[QueryMetrics] = []
    latency_bucket: List[float] = []

    for i, question in enumerate(queries, 1):
        log.info("\n--- Query %s/%s: %s", i, len(queries), question)
        with Timer() as query_timer:
            try:
                response = chain.invoke(input=question)
            except Exception as exc:  # pragma: no cover - runtime errors
                log.error("Error during query '%s': %s", question, exc)
                response = f"ERROR: {exc}"
        elapsed = query_timer.get_elapsed()
        latency_bucket.append(elapsed)
        log.info("Response time: %.2fs", elapsed)

        is_accurate = accuracy_fn(question, response) if accuracy_fn else None
        collected.append(
            QueryMetrics(
                query=question,
                response=response,
                response_time=elapsed,
                chunks_retrieved=chunks_retrieved,
                is_accurate=is_accurate,
            )
        )

    latency_summary: Dict[str, float] = {
        "min": min(latency_bucket) if latency_bucket else 0.0,
        "max": max(latency_bucket) if latency_bucket else 0.0,
        "mean": statistics.fmean(latency_bucket) if latency_bucket else 0.0,
    }
    if len(latency_bucket) > 1:
        latency_summary["stdev"] = statistics.pstdev(latency_bucket)
    return collected, latency_summary
