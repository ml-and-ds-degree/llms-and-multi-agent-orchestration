#!/usr/bin/env -S uv run

"""
Visualization utility for RAG experiments.
Generates comparison charts from JSON result files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
CHARTS_DIR = BASE_DIR / "assets"

# Map file names/keys to display labels
LABELS = {
    "experiment_1_baseline_plus": "Baseline",
    "experiment_2_llm_small": "Small LLM (1B)",
    "experiment_2_chunky": "Aggressive Chunking",
    "experiment_3_local": "Local (3B)",
    "experiment_3_hybrid": "Hybrid Cloud (120B)",
    "experiment_4_basic": "Adv. Basic",
    "experiment_4_contextual": "Contextual",
    "experiment_4_rerank": "Reranking",
}

COLORS = {
    "Baseline": "#1f77b4",
    "Small LLM (1B)": "#aec7e8",
    "Aggressive Chunking": "#ff7f0e",
    "Local (3B)": "#2ca02c",
    "Hybrid Cloud (120B)": "#98df8a",
    "Adv. Basic": "#9467bd",
    "Contextual": "#c5b0d5",
    "Reranking": "#8c564b",
}


def load_results() -> List[Dict[str, Any]]:
    """Load all JSON result files from the results directory."""
    results = []
    for file_path in RESULTS_DIR.glob("*.json"):
        # Skip files that don't match our experiment pattern or are empty
        if not file_path.stem.startswith("experiment_"):
            continue

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                # Infer label from filename
                key = file_path.stem
                data["display_label"] = LABELS.get(key, key)
                data["filename"] = key

                # Calculate average AI Score from queries if available
                queries = data.get("queries", [])
                if queries:
                    scores = [q.get("ai_score", 0.0) for q in queries]
                    data["calculated_accuracy"] = (
                        sum(scores) / len(scores) if scores else 0.0
                    )
                else:
                    data["calculated_accuracy"] = data.get("accuracy_rate", 0.0)

                results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    return results


def plot_latency_comparison(df: pd.DataFrame):
    """Generate bar chart for Average Response Time."""
    plt.figure(figsize=(12, 6))

    # Sort by latency for better readability
    df_sorted = df.sort_values("avg_response_time")

    bars = plt.bar(
        df_sorted["display_label"],
        df_sorted["avg_response_time"],
        color=[COLORS.get(l, "#7f7f7f") for l in df_sorted["display_label"]],
    )

    plt.title("Average Response Latency by Experiment", fontsize=14, pad=20)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_path = CHARTS_DIR / "latency_comparison.png"
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved latency chart to {output_path}")
    plt.close()


def plot_indexing_time(df: pd.DataFrame):
    """Generate bar chart for Indexing Time."""
    plt.figure(figsize=(12, 6))

    df_sorted = df.sort_values("indexing_time")

    bars = plt.bar(
        df_sorted["display_label"],
        df_sorted["indexing_time"],
        color=[COLORS.get(l, "#7f7f7f") for l in df_sorted["display_label"]],
    )

    plt.title("Indexing Time by Experiment", fontsize=14, pad=20)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_path = CHARTS_DIR / "indexing_time.png"
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved indexing chart to {output_path}")
    plt.close()


def plot_chunk_counts(df: pd.DataFrame):
    """Generate bar chart for Number of Chunks."""
    plt.figure(figsize=(10, 6))

    # Filter for unique chunk counts to avoid clutter if many are same
    # But for completeness, let's show all that deviate

    df_sorted = df.sort_values("num_chunks")

    bars = plt.bar(
        df_sorted["display_label"],
        df_sorted["num_chunks"],
        color=[COLORS.get(l, "#7f7f7f") for l in df_sorted["display_label"]],
    )

    plt.title("Number of Chunks Generated", fontsize=14, pad=20)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_path = CHARTS_DIR / "chunk_counts.png"
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved chunk count chart to {output_path}")
    plt.close()


def plot_accuracy_score(df: pd.DataFrame):
    """Generate bar chart for AI Accuracy Score."""
    plt.figure(figsize=(12, 6))

    # Sort by accuracy
    df_sorted = df.sort_values("calculated_accuracy", ascending=False)

    bars = plt.bar(
        df_sorted["display_label"],
        df_sorted["calculated_accuracy"],
        color=[COLORS.get(l, "#7f7f7f") for l in df_sorted["display_label"]],
    )

    plt.title("Average AI Score by Experiment", fontsize=14, pad=20)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, 1.1)  # Set y-axis to slightly above 1.0 for clarity

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_path = CHARTS_DIR / "accuracy_score.png"
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved accuracy chart to {output_path}")
    plt.close()


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    if not results:
        logger.error("No result files found in results/ directory!")
        return

    # Create DataFrame for easier plotting
    df = pd.DataFrame(results)

    # Ensure numeric columns
    df["avg_response_time"] = pd.to_numeric(df["avg_response_time"])
    df["indexing_time"] = pd.to_numeric(df["indexing_time"])
    df["num_chunks"] = pd.to_numeric(df["num_chunks"])
    # calculated_accuracy is already float from load_results

    logger.info(f"Loaded {len(df)} experiment results.")

    plot_latency_comparison(df)
    plot_indexing_time(df)
    plot_chunk_counts(df)
    plot_accuracy_score(df)

    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
