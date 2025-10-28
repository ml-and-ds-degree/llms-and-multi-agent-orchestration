# Development Container Setup

This directory contains the configuration for the development container for this project.

## Components

- **Base Image**: Debian Bookworm Slim
- **UV**: Fast Python package installer and resolver by Astral
- **Ollama**: Local LLM runtime for running large language models

## Features

- Python 3 pre-installed
- UV package manager for fast Python dependency management
- Ollama for running LLMs locally
- Port 11434 forwarded for Ollama API access
- Git and build-essential tools included

## Usage

### Using VS Code

1. Install the "Dev Containers" extension in VS Code
2. Open this repository in VS Code
3. Click on the green button in the bottom-left corner
4. Select "Reopen in Container"
5. Wait for the container to build and start

### Using UV

UV is installed and available in the container. Use it to manage Python dependencies:

```bash
# Create a new project
uv init

# Install a package
uv pip install package-name

# Install from requirements.txt
uv pip install -r requirements.txt
```

### Using Ollama

Ollama is installed and ready to use. To start using it:

```bash
# Pull a model (e.g., llama2)
ollama pull llama2

# Run a model
ollama run llama2

# List installed models
ollama list
```

The Ollama API is available on port 11434, which is forwarded from the container.

## Notes

- The container runs as root user for maximum compatibility with dev tools and to avoid permission issues
  - This is acceptable for development containers but should be changed for production use
  - The `UV_SYSTEM_PYTHON` environment variable is set to allow UV to work with system Python
- The working directory is set to `/workspace`
- All tools are pre-installed and ready to use upon container creation
- UV is pinned to version 0.4.30 for reproducibility
- Ollama installs the latest stable version (version pinning not supported by the official installer)
