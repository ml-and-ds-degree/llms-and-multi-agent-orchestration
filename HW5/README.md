# RAG Pipeline Experimentation Framework (HW5)

## Overview
This project implements and evaluates various Retrieval-Augmented Generation (RAG) pipeline architectures using Ollama. It is designed to benchmark performance metrics such as latency, accuracy, and indexing time across different configurations, including local vs. cloud deployment, model size variations, and advanced retrieval techniques like contextualization and reranking.

## Package Information
- **Package Name**: `hw5-rag-pipeline`
- **Version**: 0.1.0
- **Python**: >=3.13

## Prerequisites
*   **Devcontainer**: This project is configured to run in a devcontainer environment.
*   **Ollama**: The local inference engine must be installed and running in the devcontainer.
    *   Pull required models: `ollama pull llama3.2`, `ollama pull llama3.2:1b`, `ollama pull nomic-embed-text`
*   **Gemini API Key** (Optional): Required for calculating the "AI Score" metric.
*   **Ollama Cloud/Remote Access** (Optional): Required for the cloud/hybrid experiments.

## Installation

### Option 1: Install from Source (Development)
1.  **Navigate to HW5 directory:**
    ```bash
    cd HW5
    ```

2.  **Install with uv:**
    ```bash
    # Install with all dependencies
    uv sync
    
    # Or install with dev dependencies
    uv sync --extra dev
    ```

### Option 2: Install from requirements.txt
If you prefer using pip or other package managers:

```bash
cd HW5

# Install base dependencies
pip install -r requirements.txt

# Or install with dev dependencies
pip install -r requirements-dev.txt
```

The requirements files are generated from `pyproject.toml` using:
```bash
# Regenerate requirements.txt (if needed)
uv pip compile pyproject.toml -o requirements.txt
uv pip compile pyproject.toml --extra dev -o requirements-dev.txt
```

### Option 3: Install as a Package
```bash
# From the HW5 directory
cd HW5

# Build the package
uv build

# Install the built wheel
uv pip install dist/hw5_rag_pipeline-0.1.0-py3-none-any.whl
```

## Setup

1.  **Environment Variables:**
    Copy the example environment file and configure your keys.
    ```bash
    cp .env.example .env
    ```
    
    Edit `.env` with your specific configuration:
    *   `GEMINI_API_KEY`: API key for Google Gemini (evaluation metric).
    *   `OLLAMA_BASE_URL`: URL for the remote/cloud Ollama instance (if running Experiment 3).
    *   `OLLAMA_API_KEY`: API key for the remote/cloud instance (if required).

## Running Experiments

All experiments should be run from the `HW5` directory as Python modules:

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

## Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/test_metrics.py
uv run pytest tests/test_rag_pipeline.py
```

## Package Structure

```
HW5/
├── experiments/           # Experiment modules
│   ├── experiment_1_baseline.py
│   ├── experiment_1_baseline_plus.py
│   ├── experiment_3_local_vs_cloud.py
│   └── experiment_4_advanced.py
├── utils/                 # Core utilities
│   ├── contextualizer.py  # Contextual chunking
│   ├── metrics.py         # Performance metrics
│   ├── rag_pipeline.py    # RAG pipeline components
│   ├── rerankers.py       # Reranking algorithms
│   └── visualize.py       # Visualization tools
├── tests/                 # Test suite
├── data/                  # Input data (BOI.pdf)
├── results/               # Experiment outputs
└── queries.py            # Test queries and evaluation
```

## Using the Package in Code

```python
# Import utilities
from utils import (
    build_chain,
    build_embeddings,
    build_llm,
    ExperimentMetrics,
    Timer
)

# Import queries
from queries import get_baseline_queries, evaluate_query_accuracy

# Use in your code
queries = get_baseline_queries()
metrics = ExperimentMetrics(
    experiment_name="My Experiment",
    model_name="llama3.2"
)
```

## Results
Experiment results are saved as JSON files in `HW5/results/`. The final summary report can be found at `HW5/results/final_summary_report.md`.

## Development

To contribute or modify the package:

1. Install with dev dependencies:
   ```bash
   uv sync --extra dev
   ```

2. Run tests:
   ```bash
   uv run pytest
   ```

3. Build the package:
   ```bash
   uv build
   ```
