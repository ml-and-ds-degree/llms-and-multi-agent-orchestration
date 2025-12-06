"""
Experiment 1: Baseline RAG Implementation
Following the exact setup from the Ollama Fundamentals video.

This experiment establishes the baseline performance metrics by implementing
RAG exactly as demonstrated in the video at timestamp 2:07:29.

Video Reference: https://youtu.be/GWB9ApTPTv4?t=7649
Repository: https://github.com/pdichone/ollama-fundamentals

Configuration (Per Video):
- Model: llama3.2 (local Ollama server on http://localhost:11434)
- Embedding Model: nomic-embed-text
- Chunk Size: 1200
- Chunk Overlap: 300
- Vector Store: Chroma (persisted between runs)
- Retriever: similarity_search with k=3 (per video instructions)
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import ExperimentMetrics, Timer, print_metrics_summary
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
)
from queries import get_baseline_queries, evaluate_query_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants (EXACTLY as in video)
DOC_PATH = "./data/BOI.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIR = "./results/chroma_baseline"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300


def run_experiment():
    """Run the baseline experiment."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: BASELINE (Following Video)")
    logger.info("=" * 60)

    # Initialize metrics
    metrics = ExperimentMetrics(
        experiment_name="Experiment 1: Baseline",
        model_name=MODEL_NAME,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        persist_directory=str(Path(PERSIST_DIR).resolve()),
    )

    # Step 1: Load and process the PDF document
    try:
        data = load_documents(Path(DOC_PATH))
    except FileNotFoundError as exc:
        logger.error("Failed to load PDF: %s", exc)
        return None

    # Step 2: Split the documents into chunks
    with Timer() as chunk_timer:
        chunks = split_documents(data, CHUNK_SIZE, CHUNK_OVERLAP)
    logger.info(f"Chunking took {chunk_timer.get_elapsed():.2f}s")

    metrics.num_chunks = len(chunks)

    # Step 3: Create the vector database
    with Timer() as index_timer:
        ensure_ollama_models([MODEL_NAME, EMBEDDING_MODEL])
        embeddings = build_embeddings(EMBEDDING_MODEL)
        vector_db, _, _ = build_vector_store(
            chunks,
            embeddings,
            VECTOR_STORE_NAME,
            Path(PERSIST_DIR),
            force_reindex=True,
        )

    metrics.indexing_time = index_timer.get_elapsed()
    logger.info(f"Indexing took {metrics.indexing_time:.2f}s")

    # Step 4: Initialize the language model
    llm = build_llm(MODEL_NAME)

    # Step 5: Create the retriever
    retriever = build_retriever(vector_db, retriever_k=3)

    # Step 6: Create the chain
    chain = build_chain(retriever, llm, prompt_template=DEFAULT_PROMPT_TEMPLATE)

    # Step 7: Run test queries
    test_queries = get_baseline_queries()
    logger.info(f"\nRunning {len(test_queries)} test queries...")

    query_metrics, _ = run_query_batch(
        chain,
        test_queries,
        evaluate_query_accuracy,
        chunks_retrieved=3,
        logger_obj=logger,
    )
    metrics.queries.extend(query_metrics)

    # Print summary
    print_metrics_summary(metrics)

    # Save metrics
    output_path = "./results/experiment_1_baseline.json"
    metrics.save(output_path)

    return metrics


if __name__ == "__main__":
    # Ensure we're in the HW5 directory
    os.chdir(Path(__file__).parent.parent)

    metrics = run_experiment()

    if metrics:
        logger.info("\n✅ Baseline experiment completed successfully!")
        logger.info("Results saved to: results/experiment_1_baseline.json")
    else:
        logger.error("\n❌ Baseline experiment failed!")
