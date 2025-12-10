"""
Utilities for RAG pipeline implementation and evaluation.
"""

from .contextualizer import build_contextual_chunks
from .metrics import ExperimentMetrics, Timer, QueryMetrics, print_metrics_summary
from .rag_pipeline import (
    build_chain,
    build_embeddings,
    build_llm,
    build_retriever,
    build_vector_store,
    ensure_ollama_models,
    load_documents,
    run_query_batch,
    split_documents,
    DEFAULT_PROMPT_TEMPLATE,
)
from .rerankers import RerankResult, llm_rerank
from .visualize import (
    load_results,
    plot_latency_comparison,
    plot_indexing_time,
    plot_chunk_counts,
    plot_accuracy_score,
)

__all__ = [
    # Contextualizer
    "build_contextual_chunks",
    # Metrics
    "ExperimentMetrics",
    "Timer",
    "QueryMetrics",
    "print_metrics_summary",
    # RAG Pipeline
    "build_chain",
    "build_embeddings",
    "build_llm",
    "build_retriever",
    "build_vector_store",
    "ensure_ollama_models",
    "load_documents",
    "run_query_batch",
    "split_documents",
    "DEFAULT_PROMPT_TEMPLATE",
    # Rerankers
    "RerankResult",
    "llm_rerank",
    # Visualization
    "load_results",
    "plot_latency_comparison",
    "plot_indexing_time",
    "plot_chunk_counts",
    "plot_accuracy_score",
]
