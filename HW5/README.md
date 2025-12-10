# RAG Pipeline Experimentation Framework (HW5)

## Overview
This project implements and evaluates various Retrieval-Augmented Generation (RAG) pipeline architectures using Ollama. It is designed to benchmark performance metrics such as latency, accuracy, and indexing time across different configurations, including local vs. cloud deployment, model size variations, and advanced retrieval techniques like contextualization and reranking.

## Prerequisites
*   **Devcontainer**: This project is configured to run in a devcontainer environment.
*   **Ollama**: The local inference engine must be installed and running in the devcontainer.
    *   Pull required models: `ollama pull llama3.2`, `ollama pull llama3.2:1b`, `ollama pull nomic-embed-text`
*   **Gemini API Key** (Optional): Required for calculating the "AI Score" metric.
*   **Ollama Cloud/Remote Access** (Optional): Required for the cloud/hybrid experiments.

## Setup

1.  **Open in Devcontainer:**
    Open this repository in VS Code and use "Reopen in Container" to start the devcontainer environment.

2.  **Install Dependencies:**
    This project uses `uv` for dependency management. Simply run:
    ```bash
    uv sync
    ```

3.  **Environment Variables:**
    Copy the example environment file and configure your keys.
    ```bash
    cp .env.example .env
    ```
    
    Edit `.env` with your specific configuration:
    *   `GEMINI_API_KEY`: API key for Google Gemini (evaluation metric).
    *   `OLLAMA_BASE_URL`: URL for the remote/cloud Ollama instance (if running Experiment 3).
    *   `OLLAMA_API_KEY`: API key for the remote/cloud instance (if required).

## Running Experiments

Navigate to the `HW5` directory:
```bash
cd HW5
```

### 1. Baseline & Baseline Plus
Evaluates the standard local RAG setup.
```bash
python -m experiments.experiment_1_baseline
python -m experiments.experiment_1_baseline_plus
```

### 2. Architectural Deviations
Tests smaller models and different chunking strategies.
```bash
python -m experiments.experiment_2_llm_small
python -m experiments.experiment_2_chunky
```

### 3. Local vs. Cloud
Compares local inference against a cloud-hosted model.
*Requires `OLLAMA_BASE_URL` to be set in `.env`.*
```bash
python -m experiments.experiment_3_local_vs_cloud
```

### 4. Advanced Retrieval
Tests contextual retrieval and reranking techniques.
```bash
python -m experiments.experiment_4_advanced
```

## Results
Experiment results are saved as JSON files in `HW5/results/`. The final summary report can be found at `HW5/results/final_summary_report.md`.
