# =============================================================================
# Travel Order Resolver - Production Dockerfile (GPU)
# =============================================================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

# Add poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (main + ml for GPU inference)
RUN poetry install --only main --with ml --no-interaction --no-ansi

# Download spaCy transformer model (benefits from GPU)
RUN python -m spacy download fr_dep_news_trf

# Copy application code
COPY src/ ./src/
COPY datasets/raw/sncf/ ./datasets/raw/sncf/

# Default command: run CLI in interactive mode
CMD ["python", "-m", "src.main", "--interactive"]
