#!/bin/bash
# Setup script for testing environment

set -e

echo "🧪 Setting up testing environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "📦 Installing dev dependencies..."
uv sync --extra dev

echo "🌐 Installing Playwright browsers..."
uv run playwright install chromium
playwright install-deps

echo ""
echo "✅ Testing environment setup complete!"
echo ""
echo "Quick start commands:"
echo "  - Run all tests:        uv run pytest"
echo "  - Run unit tests:       uv run pytest -m unit"
echo "  - Run E2E tests:        uv run pytest -m e2e"
echo "  - Run with coverage:    uv run pytest --cov=app"
echo ""
echo "📖 See tests/README.md for detailed documentation"
