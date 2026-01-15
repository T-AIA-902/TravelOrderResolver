# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] Integration finetuning CamemBERT - 2025-01-15

### Added
- **CamemBERT Entity Extractor** (`src/nlp/entity_extractor.py`):
  - `CamembertEntityExtractor` - Fine-tuned CamemBERT NER model
  - Direct DEP/DEST label prediction (no heuristics needed)
- **Pathfinding module** (`src/pathfinding/`):
  - `TrainGraph` - NetworkX-based railway graph with Dijkstra algorithm
  - LGV (high-speed line) optimization (3x faster weight)
  - City hub transfers for major stations (Paris, Lyon, etc.)
- **Visualization module** (`src/visualization/`):
  - `MapVisualizer` - Folium-based interactive route maps
  - Auto-opens in browser after generation
- **Main orchestrator** (`src/main.py`):
  - `TravelOrderResolver` - Unified entry point with pluggable extractors
  - CLI with `--extractor` flag: camembert, spacy, fuzzy, regex
- **Saved model** (`models/camembert-ner/`):
  - Pre-fine-tuned CamemBERT tokenizer and config
- **Dataset generation scripts** (`datasets/scripts/`):
  - `generate_camembert_data.py` - 100K synthetic training samples
  - `preprocess_camembert_data.py` - CSV to JSONL conversion
- **Training dataset** (`datasets/processed/dataset_train_sncf.csv`):
  - 100,000 labeled travel sentences (80% valid, 20% invalid)
- **Dependencies**:
  - networkx, shapely, geopy (pathfinding)
  - folium (visualization)
  - faker (data generation)

## [0.1.2] Integration SpaCy - 2025-01-15

### Added
- **SpaCy Entity Extractor** (`src/nlp/entity_extractor.py`):
  - `SpacyEntityExtractor` - Extraction NER avec fr_core_news_lg
  - `FuzzyEntityExtractor` - Extension avec matching fuzzy RapidFuzz
- **Fuzzy Matcher** (`src/nlp/fuzzy_matcher.py`):
  - `StationMatcher` - Matching fuzzy avec RapidFuzz
  - Utilise `StationDatabase` comme source de données unique
- **Evaluation** (`evaluation/`):
  - `metrics.py` - Calcul precision/recall/F1/accuracy
  - `evaluate_spacy.py` - Évaluation extracteur SpaCy
  - `evaluate_fuzzy.py` - Évaluation extracteur fuzzy
- **Tests** (`tests/unit/`):
  - `test_entity_extractor.py` - 27 tests extracteurs
  - `test_fuzzy_matcher.py` - 26 tests fuzzy matcher

### Changed
- **Data module** - `normalize_name()` utilise maintenant `unidecode` pour une normalisation plus robuste des accents
- **Evaluation scripts** - Utilisent `datasets/generated/test.csv` (1501 phrases) au lieu d'un dataset séparé

## [0.1.1] - 2025-01-15

### Added
- Initial project structure
- README with documentation
- Poetry configuration (pyproject.toml)
- Makefile with common commands
- GitHub Actions CI/CD templates
- Pre-commit hooks configuration
- **Data module** (`src/data/`):
  - `DataLoader` class for loading SNCF JSON files
  - `StationDatabase` class with 6,101 stations
  - `normalize_name()` function for text normalization
  - City-to-station mapping and aliases support
- **NLP module** (`src/nlp/`):
  - `BaseModel` abstract interface
  - `Preprocessor` with unicode normalization and tokenization
  - `BaselineRegexModel` for rule-based extraction
  - `NLPPipeline` for orchestrating NLP processing
  - Intent classification: TRIP, NOT_TRIP, NOT_FRENCH, UNKNOWN
  - Entity extraction: DEPARTURE, DESTINATION, INTERMEDIATE
- **Dataset generation** (`datasets/scripts/`):
  - `generate_sentences.py` - 10,000 training sentences
  - `validate_dataset.py` - dataset validation
  - `import_sncf_data.py` - SNCF data import CLI
- **Unit tests** - 63 tests covering data and NLP modules

### Changed
- Updated README to reflect the final architecture

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- Test discovery issue: added missing `tests/unit/__init__.py`
- `.gitignore` pattern `models/` changed to `/models/` to include `src/nlp/models/` source code

### Security
- N/A

---

## [0.1.0] - YYYY-MM-DD

### Added
- [TODO] Baseline regex model
- [TODO] NLP pipeline
- [TODO] Pathfinding with Dijkstra
- [TODO] CLI interface
- [TODO] Dataset generation scripts

---

*Format: [version] - date*
