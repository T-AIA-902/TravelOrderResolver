.PHONY: install install-dev install-ml test lint format clean run help

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# INSTALLATION
# =============================================================================

install: ## Install production dependencies
	poetry install --only main
	poetry run python -m spacy download fr_dep_news_trf

install-dev: ## Install development dependencies
	poetry install
	poetry run python -m spacy download fr_dep_news_trf
	poetry run pre-commit install

install-ml: ## Install ML/training dependencies
	poetry install --with ml
	poetry run python -m spacy download fr_dep_news_trf

install-all: ## Install all dependencies (dev + ml + notebooks)
	poetry install --with dev,ml,notebooks
	poetry run python -m spacy download fr_dep_news_trf
	poetry run pre-commit install

# =============================================================================
# DEVELOPMENT
# =============================================================================

dev-setup: install-dev ## Full development setup
	@echo "Development environment ready!"

test: ## Run tests with coverage
	poetry run pytest

test-unit: ## Run unit tests only
	poetry run pytest tests/unit -v

test-integration: ## Run integration tests only
	poetry run pytest tests/integration -v

test-e2e: ## Run end-to-end tests only
	poetry run pytest tests/e2e -v

lint: ## Run all linters
	poetry run flake8 src tests
	poetry run mypy src
	poetry run isort --check-only src tests
	poetry run black --check src tests

format: ## Format code with black and isort
	poetry run isort src tests
	poetry run black src tests

typecheck: ## Run type checking
	poetry run mypy src

# =============================================================================
# RUNNING
# =============================================================================

run: ## Run the main CLI
	poetry run python -m src.main

run-interactive: ## Run in interactive mode
	poetry run python -m src.main --interactive

run-api: ## Start the API server
	poetry run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

run-demo: ## Launch Gradio demo (if available)
	poetry run python -m src.api.demo

# =============================================================================
# NLP & TRAINING
# =============================================================================

train-baseline: ## Train baseline model
	poetry run python training/train_classifier.py --model baseline

train-camembert: ## Fine-tune CamemBERT
	poetry run python training/fine_tune_camembert.py

train-flan: ## Fine-tune Flan-T5
	poetry run python training/fine_tune_flan_t5.py

benchmark: ## Run model benchmark
	poetry run python evaluation/benchmark.py

# =============================================================================
# DATA
# =============================================================================

download-data: ## Download SNCF data
	poetry run python datasets/scripts/import_sncf_data.py

generate-dataset: ## Generate training dataset
	poetry run python datasets/scripts/generate_sentences.py

augment-data: ## Augment dataset
	poetry run python datasets/scripts/augment_data.py

validate-dataset: ## Validate dataset
	poetry run python datasets/scripts/validate_dataset.py

# =============================================================================
# MODELS
# =============================================================================

download-models: ## Download pre-trained models
	./scripts/download_models.sh

# =============================================================================
# DOCKER
# =============================================================================

docker-build: ## Build Docker image
	docker build -t travel-order-resolver -f docker/Dockerfile .

docker-run: ## Run Docker container
	docker run -p 8000:8000 travel-order-resolver

docker-compose-up: ## Start all services with docker-compose
	docker-compose -f docker/docker-compose.yml up -d

docker-compose-down: ## Stop all services
	docker-compose -f docker/docker-compose.yml down

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs: ## Generate documentation
	@echo "Documentation generation not yet implemented"

# =============================================================================
# CLEANUP
# =============================================================================

clean: ## Clean up cache and build files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf dist build .eggs

clean-models: ## Clean downloaded models (use with caution)
	rm -rf models/checkpoints/*
	rm -rf models/experiments/*

# =============================================================================
# HELP
# =============================================================================

help: ## Show this help message
	@echo "Travel Order Resolver - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
