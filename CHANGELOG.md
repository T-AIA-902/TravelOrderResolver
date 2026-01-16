# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] Modular NLP Architecture & Performance - 2025-01-16

### Added
- **Modular NLP Architecture** (`src/nlp/`):
  - `interfaces.py` - ABC classes: `IntentClassifier`, `EntityExtractor`, `PostProcessor`
  - `intent/` - Intent classifiers: `RegexIntentClassifier`, `CamembertIntentClassifier`
  - `entity/` - Entity extractors: `RegexEntityExtractor`, `SpacyEntityExtractor`, `CamembertEntityExtractor`
  - `post/` - Post-processors: `FuzzyPostProcessor`
- **Batched Inference**:
  - `CamembertIntentClassifier.classify_batch()` - GPU-efficient batching (batch_size=128)
  - `SpacyEntityExtractor.extract_batch()` - Uses `nlp.pipe()` for batched processing
- **FuzzyMatcher Optimizations** (`src/nlp/fuzzy_matcher.py`):
  - Pre-built `first_words` index in `__init__` (computed once, reused for all queries)
  - In-memory LRU cache `_match_cache` for repeated queries
  - `clear_cache()` and `get_cache_stats()` methods
  - `process_batch()` method for batch processing
- **Short Dataset Splits** (`datasets/splits/short-splits/`):
  - `test.csv` (1,500 samples), `train.csv` (7,000 samples), `val.csv` (1,500 samples)
  - Faster evaluation iterations during development
- **Unified Evaluation Script** (`evaluation/evaluate_all.py`):
  - 5 evaluation tables: Intent, Entity, Entity+Fuzzy, Combined, Combined+Fuzzy
  - Model caching at startup (avoid re-initializing CamemBERT 4x)
  - JSON export with `--output-json` flag

### Changed
- **Data Generator Fix** (`datasets/scripts/generate_camembert_data.py`):
  - Fixed phantom departure bug: dest-only templates now correctly have `departure=None`
  - Split templates into `structures_both` and `structures_dest_only`
  - Uses `StationDatabase` instead of CSV file
- **README.md**: Updated with 5 evaluation tables and modular architecture
- **TASKS.md**: Added fuzzy threshold evaluation and dest-only metrics TODOs

### Performance
- Evaluation time reduced from 60+ min to ~10 min (model caching + batching)
- FuzzyMatcher speedup via caching (repeated queries hit cache)

### Results on short-splits/test.csv (1,500 samples)
| Pipeline | Intent Acc | Entity Acc | Latency |
|----------|------------|------------|---------|
| Regex + CamemBERT + Fuzzy | 71.9% | 28.6% | 1.4ms |

## [0.1.5] CamemBERT Zero-Shot Baseline - 2025-01-16

### Added
- **CamemBERT Zero-Shot Extractor** (`src/nlp/entity_extractor.py`):
  - `CamembertZeroShotExtractor` - Uses `almanach/camembert-base` without fine-tuning
  - Exact string matching against station database
  - Establishes baseline for future fine-tuning comparison
- **Evaluation Adapters** (`evaluation/evaluate_all.py`):
  - `CamembertAdapter` - Entity extraction only
  - `CamembertRegexAdapter` - Combined with Regex for intent classification
  - Updated evaluation tables to include CamemBERT results

### Changed
- **README.md**: Added CamemBERT metrics to benchmark tables
- **.flake8**: Added `extend-ignore = E203` for black compatibility

### Results on test.csv (15k samples)
| Model | Entity Accuracy | Intent Accuracy | Latency |
|-------|-----------------|-----------------|---------|
| CamemBERT (zero-shot) | 6.7% | N/A | 1.2ms |
| CamemBERT + Regex | 6.7% | 69.3% | 2.2ms |

## [0.1.4] Unified Evaluation & Metrics - 2025-01-15

### Added
- **Unified Evaluation Script** (`evaluation/evaluate_all.py`):
  - Evaluates all NLP methods (Regex, SpaCy, Fuzzy) with consistent methodology
  - Supports both JSON and CSV dataset formats
  - Generates 3 metrics tables: entity extractors, complete solutions, categories
  - Category-specific metrics: misspelling handling, no-caps handling, order detection
  - Export results to JSON
- **Dataset Split Script** (`datasets/scripts/split_dataset.py`):
  - Splits dataset into train/val/test (70/15/15)
  - Reproducible with seed parameter
- **Dataset Splits** (`datasets/splits/`):
  - `train.csv` (70,000 samples)
  - `val.csv` (15,000 samples)
  - `test.csv` (15,000 samples)

### Changed
- **README.md**: Updated benchmarks section with actual metrics from evaluation
- **evaluate_all.py**: Added support for VALID/INVALID label mapping to TRIP/NOT_TRIP

### Results on test.csv (15k samples)
| Model | Entity Accuracy | Intent Accuracy | Latency |
|-------|-----------------|-----------------|---------|
| Baseline Regex | 17.8% | 69.3% | 1.7ms |
| SpaCy | 6.6% | N/A | 7.5ms |
| Fuzzy | 11.5% | N/A | 22.9ms |

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
