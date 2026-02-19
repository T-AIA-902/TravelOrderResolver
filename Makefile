.PHONY: install install-dev install-ml test lint format clean run help evaluate evaluate-full demo demo-camembert demo-spacy demo-regex demo-all front-build front-run front-stop front-rebuild

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

# =============================================================================
# DEMOS (Interactive with map visualization)
# =============================================================================

demo: demo-camembert ## Run demo with default (CamemBERT) extractor

demo-camembert: ## Demo with CamemBERT extractor
	@echo "Demo: CamemBERT extractor"
	echo "1,Je veux aller de Paris a Lyon" | poetry run python -m src.main --extractor camembert

demo-spacy: ## Demo with SpaCy extractor
	@echo "Demo: SpaCy extractor"
	echo "1,Je veux aller de Paris a Lyon" | poetry run python -m src.main --extractor spacy

demo-regex: ## Demo with Regex extractor
	@echo "Demo: Regex extractor"
	echo "1,Je veux aller de Paris a Lyon" | poetry run python -m src.main --extractor regex

demo-all: demo-regex demo-spacy demo-camembert ## Run demo with all extractors

# =============================================================================
# NLP & EVALUATION
# =============================================================================

evaluate: ## Run NLP evaluation (5 tables, augmented dataset)
	poetry run python -m src.evaluation --eval-type all --dataset datasets/augmented/test.csv

evaluate-full: ## Run NLP evaluation (5 tables, full augmented dataset)
	poetry run python -m src.evaluation --eval-type all --dataset datasets/augmented/stt_dataset_100k.csv

# =============================================================================
# DATA
# =============================================================================

download-data: ## Download SNCF data
	poetry run python datasets/scripts/import_sncf_data.py

generate-dataset: ## Generate base dataset (clean, no STT errors)
	poetry run python datasets/scripts/generate_base.py

augment-data: ## Apply STT augmentation to base dataset
	poetry run python datasets/scripts/augment_stt.py

validate-dataset: ## Validate dataset
	poetry run python datasets/scripts/validate_dataset.py --input datasets/augmented

# =============================================================================
# MODELS
# =============================================================================

download-models: ## Download pre-trained models
	./scripts/download_models.sh

# =============================================================================
# DOCKER
# =============================================================================

docker-build: ## Build backend Docker image
	docker build -t travel-order-resolver -f docker/Dockerfile .

docker-run: ## Run backend Docker container
	docker run -p 8000:8000 travel-order-resolver

docker-compose-up: ## Start all services with docker-compose
	docker-compose -f docker/docker-compose.yml up -d

docker-compose-down: ## Stop all services
	docker-compose -f docker/docker-compose.yml down

# =============================================================================
# FRONTEND
# =============================================================================

front-build: ## Build frontend Docker image
	docker build -f docker/Dockerfile.frontend -t tor-frontend .

front-run: front-build ## Build and run frontend on http://localhost:3000
	@docker rm -f tor-frontend 2>/dev/null || true
	docker run -d --name tor-frontend -p 3000:80 tor-frontend
	@echo "Frontend running on http://localhost:3000"

front-stop: ## Stop frontend container
	docker stop tor-frontend

front-rebuild: front-run ## Rebuild and restart frontend

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
