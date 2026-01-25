.PHONY: install install-dev install-ml test lint format clean run help evaluate evaluate-full demo demo-camembert demo-spacy demo-regex demo-all colab-notebook install-adapter test-unified eval-unified

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
# TRAINING (QLoRA Fine-tuning)
# =============================================================================

# NOTE: Local training requires ~9.73 GB VRAM (RTX 2060 6GB is insufficient)
# Use Google Colab for training: training/notebooks/ministral_unified_colab.ipynb

train-ministral-intent: ## Train Ministral for intent classification (requires ~10GB VRAM)
	poetry run python -m training.scripts.train_ministral_intent --config training/config/ministral_intent.yaml

train-ministral-entity: ## Train Ministral for entity extraction (requires ~10GB VRAM)
	poetry run python -m training.scripts.train_ministral_entity --config training/config/ministral_entity.yaml

train-ministral-all: train-ministral-intent train-ministral-entity ## Train both Ministral models

merge-lora-intent: ## Merge intent LoRA adapter with base model
	poetry run python -m training.scripts.merge_lora_weights \
		--adapter-path models/ministral-intent-lora \
		--output-dir models/ministral-intent-merged

merge-lora-entity: ## Merge entity LoRA adapter with base model
	poetry run python -m training.scripts.merge_lora_weights \
		--adapter-path models/ministral-entity-lora \
		--output-dir models/ministral-entity-merged

# =============================================================================
# MINISTRAL UNIFIED (Train on Colab, evaluate locally)
# =============================================================================

colab-notebook: ## Open Colab notebook URL for unified training
	@echo "Open this notebook in Google Colab for training:"
	@echo "https://colab.research.google.com/github/$(shell git remote get-url origin | sed 's/.*github.com[:/]//;s/.git$$//')/blob/$(shell git branch --show-current)/training/notebooks/ministral_unified_colab.ipynb"
	@echo ""
	@echo "After training, download ministral-unified-lora.zip and run:"
	@echo "  make install-adapter"

install-adapter: ## Install downloaded LoRA adapter from Colab
	@if [ -f ministral-unified-lora.zip ]; then \
		unzip -o ministral-unified-lora.zip -d models/; \
		echo "Adapter installed to models/ministral-unified-lora/"; \
	else \
		echo "Error: ministral-unified-lora.zip not found"; \
		echo "Download it from Colab first."; \
		exit 1; \
	fi

test-unified: ## Test unified Ministral model inference
	@if [ -d models/ministral-unified-lora ]; then \
		poetry run python -c "from src.nlp.unified import MinistralUnifiedNLP; \
		nlp = MinistralUnifiedNLP('models/ministral-unified-lora'); \
		print('Test 1:', nlp.process('Je voudrais aller de Paris a Lyon')); \
		print('Test 2:', nlp.process('Quel temps fait-il demain?'))"; \
	else \
		echo "Error: Adapter not found. Run 'make install-adapter' first."; \
		exit 1; \
	fi

eval-unified: ## Evaluate unified Ministral model
	poetry run python -m src.evaluation --models ministral-unified --eval-type all

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
