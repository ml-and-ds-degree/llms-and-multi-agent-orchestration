"""
Metrics tracking utility for RAG experiments.
Tracks timing, accuracy, and performance metrics.
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class QueryMetrics:
    """Metrics for a single query."""

    query: str
    response: str
    response_time: float
    chunks_retrieved: int
    is_accurate: Optional[bool] = None  # Will be manually assessed


@dataclass
class ExperimentMetrics:
    """Metrics for an entire experiment."""

    experiment_name: str
    model_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    persist_directory: Optional[str] = None
    indexing_time: float = 0.0
    num_chunks: int = 0
    queries: List[QueryMetrics] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pass

    @property
    def avg_response_time(self) -> float:
        """Calculate average response time across all queries."""
        if not self.queries:
            return 0.0
        return sum(q.response_time for q in self.queries) / len(self.queries)

    @property
    def accuracy_rate(self) -> float:
        """Calculate accuracy rate (only for queries with assessed accuracy)."""
        assessed = [q for q in self.queries if q.is_accurate is not None]
        if not assessed:
            return 0.0
        accurate = sum(1 for q in assessed if q.is_accurate)
        return accurate / len(assessed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["avg_response_time"] = self.avg_response_time
        data["accuracy_rate"] = self.accuracy_rate
        return data

    def save(self, output_path: str):
        """Save metrics to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Metrics saved to: {output_path}")


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed if self.elapsed is not None else 0.0


def print_metrics_summary(metrics: ExperimentMetrics):
    """Print a formatted summary of experiment metrics."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY: {metrics.experiment_name}")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  - Model: {metrics.model_name}")
    print(f"  - Embedding Model: {metrics.embedding_model}")
    print(f"  - Chunk Size: {metrics.chunk_size}")
    print(f"  - Chunk Overlap: {metrics.chunk_overlap}")
    print(f"  - Persist Directory: {metrics.persist_directory or 'None (in-memory)'}")

    print("\nIndexing Metrics:")
    print(f"  - Indexing Time: {metrics.indexing_time:.2f}s")
    print(f"  - Number of Chunks: {metrics.num_chunks}")

    print("\nQuery Metrics:")
    print(f"  - Total Queries: {len(metrics.queries)}")
    print(f"  - Average Response Time: {metrics.avg_response_time:.2f}s")

    if any(q.is_accurate is not None for q in metrics.queries):
        print(f"  - Accuracy Rate: {metrics.accuracy_rate:.1%}")

    print("\nQuery Details:")
    for i, q in enumerate(metrics.queries, 1):
        print(f"\n  Query {i}: {q.query}")
        print(f"    - Response Time: {q.response_time:.2f}s")
        print(f"    - Chunks Retrieved: {q.chunks_retrieved}")
        if q.is_accurate is not None:
            print(f"    - Accurate: {'Yes' if q.is_accurate else 'No'}")
        print(f"    - Response: {q.response[:100]}...")

    print("\n" + "=" * 60 + "\n")
