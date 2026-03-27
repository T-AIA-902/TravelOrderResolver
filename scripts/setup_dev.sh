#!/bin/bash
# =============================================================================
# Travel Order Resolver - Development Setup Script
# =============================================================================

set -e

echo "Setting up development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi

echo "Python version: $python_version"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Poetry version: $(poetry --version)"

# Install dependencies
echo "Installing dependencies..."
poetry install

# Download spaCy model
echo "Downloading spaCy French model..."
poetry run python -m spacy download fr_core_news_lg

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
poetry run pre-commit install

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

echo ""
echo "Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your configuration"
echo "  2. Run 'make download-data' to get SNCF data"
echo "  3. Run 'make test' to verify installation"
echo ""
