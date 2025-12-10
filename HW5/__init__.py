"""
HW5: RAG Pipeline Experimentation Framework

This package provides tools and utilities for building, testing, and evaluating
Retrieval-Augmented Generation (RAG) pipelines using Ollama.
"""

__version__ = "0.1.0"

from .queries import (
    get_baseline_queries,
    get_all_queries,
    get_query_info,
    evaluate_query_accuracy,
)

__all__ = [
    "get_baseline_queries",
    "get_all_queries",
    "get_query_info",
    "evaluate_query_accuracy",
]
