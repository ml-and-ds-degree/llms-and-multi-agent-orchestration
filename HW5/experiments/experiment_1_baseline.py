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

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

from utils.metrics import ExperimentMetrics, QueryMetrics, Timer, print_metrics_summary
from queries import get_baseline_queries, get_query_info, evaluate_query_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants (EXACTLY as in video)
DOC_PATH = "./data/BOI.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIR = "./results/chroma_baseline"
PORT = 11434
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300


def ingest_pdf(doc_path: str):
    """Load PDF documents."""
    logger.info(f"Loading PDF from: {doc_path}")
    if os.path.exists(doc_path):
        loader = PyPDFLoader(file_path=doc_path)
        data = loader.load()
        logger.info("PDF loaded successfully.")
        return data
    else:
        logger.error(f"PDF file not found at path: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    logger.info(
        f"Splitting documents with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}"
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documents split into {len(chunks)} chunks.")
    return chunks


def create_vector_db(chunks):
    """Create a vector database from document chunks."""
    logger.info("Validating Ollama server availability on localhost:%s", PORT)
    try:
        ollama.list()
    except Exception as exc:
        logger.error("Ollama server not reachable: %s", exc)
        raise

    logger.info(f"Pulling embedding model: {EMBEDDING_MODEL}")
    ollama.pull(EMBEDDING_MODEL)

    logger.info("Creating vector database with persistence at %s", PERSIST_DIR)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
        persist_directory=PERSIST_DIR,
    )
    logger.info("Vector database created and persisted.")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a retriever with k=3 (exactly as described in the baseline video)."""
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    logger.info("Retriever created (similarity_search k=3).")
    return retriever


def create_chain(retriever, llm):
    """Create the RAG chain (EXACTLY as in video)."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("Chain created successfully.")
    return chain


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
    data = ingest_pdf(DOC_PATH)
    if data is None:
        logger.error("Failed to load PDF. Exiting.")
        return None

    # Step 2: Split the documents into chunks
    with Timer() as chunk_timer:
        chunks = split_documents(data)
    logger.info(f"Chunking took {chunk_timer.get_elapsed():.2f}s")

    metrics.num_chunks = len(chunks)

    # Step 3: Create the vector database
    with Timer() as index_timer:
        vector_db = create_vector_db(chunks)

    metrics.indexing_time = index_timer.get_elapsed()
    logger.info(f"Indexing took {metrics.indexing_time:.2f}s")

    # Step 4: Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Step 5: Create the retriever
    retriever = create_retriever(vector_db, llm)

    # Step 6: Create the chain
    chain = create_chain(retriever, llm)

    # Step 7: Run test queries
    test_queries = get_baseline_queries()
    logger.info(f"\nRunning {len(test_queries)} test queries...")

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n--- Query {i}/{len(test_queries)} ---")
        logger.info(f"Question: {query}")

        with Timer() as query_timer:
            try:
                response = chain.invoke(input=query)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                response = f"ERROR: {str(e)}"

        query_time = query_timer.get_elapsed()
        logger.info(f"Response time: {query_time:.2f}s")
        logger.info(f"Response: {response[:200]}...")

        # Record metrics
        is_accurate = evaluate_query_accuracy(query, response)
        query_info = get_query_info(query)
        retrieved_chunks = 3

        if query_info is None:
            logger.warning("Query metadata missing for %s; accuracy set to None", query)

        if is_accurate is None:
            logger.warning("Could not auto-assess accuracy for query '%s'", query)

        query_metrics = QueryMetrics(
            query=query,
            response=response,
            response_time=query_time,
            chunks_retrieved=retrieved_chunks,
            is_accurate=is_accurate,
        )
        metrics.queries.append(query_metrics)

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
