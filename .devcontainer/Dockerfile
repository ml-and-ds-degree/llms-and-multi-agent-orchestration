# Use Debian slim as the base image
FROM debian:bookworm-slim

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install UV by Astral
# UV is a fast Python package installer and resolver
# Using official installation script with specific version for reproducibility
ENV UV_VERSION=0.4.30
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh

# Add UV to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Ollama
# Using official installation script
# Note: Ollama doesn't provide version-specific installation URLs in their script
# The script installs the latest stable version
RUN curl -fsSL https://ollama.com/install.sh | sh

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["/bin/bash"]
