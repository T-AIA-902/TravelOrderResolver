# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] GPU Support & Docker Updates - 2025-01-22

### Added
- **Device Utility Module** (`src/utils/device.py`):
  - `get_torch_device(preferred)` - Auto-detect PyTorch device (cuda/cpu)
  - `setup_spacy_device(preferred)` - Configure SpaCy GPU via `prefer_gpu()`
  - `get_device_info()` - Get device diagnostics (GPU name, CUDA availability)
- **GPU Support for CamemBERT Models**:
  - `CamembertIntentClassifier`: Added `device` parameter, passes to HuggingFace pipeline
  - `CamembertEntityExtractor`: Added `device` parameter, moves model and tensors to GPU
- **GPU Support for SpaCy**:
  - `SpacyEntityExtractor`: Added `device` parameter, calls `spacy.prefer_gpu()` before loading
- **Evaluation CLI `--device` Flag**:
  - New argument: `--device {auto,cuda,cpu}` (default: auto)
  - Displays GPU info at startup when available

### Changed
- **Docker Files Updated**:
  - `docker/Dockerfile`: GPU base image (`pytorch:2.1.0-cuda12.1`), `fr_dep_news_trf` model, CLI entry point
  - `docker/Dockerfile.training`: Updated poetry install command
  - `docker/docker-compose.yml`: Removed dead API service, added `evaluate` service, GPU reservations

### Architecture
Per-model device selection with `device="auto"`:

| Model Type | GPU Support | Auto-Detection |
|------------|-------------|----------------|
| Regex (intent/entity) | N/A | Pure Python, always CPU |
| Langdetect | N/A | Pure Python, always CPU |
| CamemBERT (intent/entity) | Yes | `torch.cuda.is_available()` |
| SpaCy | Yes | `spacy.prefer_gpu()` |

### Dependencies
- PyTorch 2.5.1+cu121 (CUDA 12.1 support)
- SpaCy GPU requires: `pip install spacy[cuda12x]`

---

## [0.3.2] Multilingual Support & Langdetect - 2025-01-22

### Added
- **Langdetect Language Detector** (`src/nlp/language/langdetect_language.py`):
  - `LangdetectLanguageDetector` - Library-based detection using `langdetect`
  - Maps ISO codes: `fr` → FRENCH, `en` → ENGLISH, others → UNKNOWN
  - 73.8% accuracy (vs Regex 68.9%), 9.69ms latency
  - Seed set for reproducibility (`DetectorFactory.seed = 0`)
- **English Entity Extraction Patterns** (`src/nlp/entity/regex_entity.py`):
  - `from_to_pattern`: Matches "from X to Y" (+ variants: towards, for)
  - `x_to_y_pattern`: Matches simple "X to Y"
  - `en_via_pattern`: Matches "via", "through", "stopping at"
  - English stopwords added to `_find_potential_stations()`
  - English articles handled in `_clean_station_name()` (the, a, an)
- **English Entity Extraction Tests** (`tests/unit/test_nlp.py`):
  - `TestRegexEntityExtractorEnglish` class with 6 tests

### Changed
- **Intent Enum** (`src/nlp/types.py`):
  - Removed `NOT_FRENCH` value (was mixing language detection with intent)
  - Intent now cleanly separates from language: `TRIP`, `NOT_TRIP`, `UNKNOWN`
- **Pipeline** (`src/nlp/pipeline.py`):
  - Removed early return for non-French text
  - Intent classification now runs regardless of detected language
- **Evaluation CLI** (`src/evaluation/cli.py`):
  - Added `langdetect` to `--models` choices
  - Removed `combined` and `combined_fuzzy` from `--eval-type all` (too slow)
  - Combined evaluations now only run when explicitly requested
- **Dependencies** (`pyproject.toml`):
  - Added `langdetect = "^1.0.9"`

### Performance Improvements
Entity extraction accuracy improved with English patterns:

| Model | Fuzzy | Before | After | Improvement |
|-------|-------|--------|-------|-------------|
| Regex | - | 29.2% | 32.9% | +3.7pp |
| Regex | ✓ | 51.3% | 56.8% | +5.5pp |

### Architecture
Consistent multilingual support across all components:

| Component | French | English |
|-----------|--------|---------|
| LanguageDetector | ✓ | ✓ |
| IntentClassifier | ✓ | ✓ |
| EntityExtractor | ✓ | ✓ (NEW) |

---

## [0.3.1] Legacy Code Removal & Pipeline Refactoring - 2025-01-22

### Added
- **Types Module** (`src/nlp/types.py`):
  - `Intent` enum (TRIP, NOT_TRIP, UNKNOWN) - Note: NOT_FRENCH removed in 0.3.2
  - `Language` enum (FRENCH, ENGLISH, UNKNOWN)
  - `TravelEntity` dataclass
  - `PredictionResult` dataclass
  - Extracted from deleted `base_model.py` for reuse

### Changed
- **`src/nlp/pipeline.py`** - Refactored to compose modular components:
  - Now uses `RegexLanguageDetector`, `RegexIntentClassifier`, `RegexEntityExtractor`
  - Supports dependency injection for custom components
  - Removed dependency on `BaselineRegexModel`
- **`src/nlp/__init__.py`** - Removed legacy model exports (`BaseModel`, `BaselineRegexModel`)
- **`tests/unit/test_nlp.py`** - Migrated tests from `BaselineRegexModel` to modular components

### Removed
- **Legacy Monolithic Model** (~643 lines):
  - `src/nlp/models/` - Entire directory deleted
  - `src/nlp/models/baseline_regex.py` (484 lines) - Combined lang+intent+entity in one class
  - `src/nlp/models/base_model.py` (159 lines) - ABC and types (moved to `types.py`)

### Architecture
```
src/nlp/
├── entity/           # 3 extractors: Regex, SpaCy, CamemBERT
├── intent/           # 2 classifiers: Regex, CamemBERT
├── language/         # 1 detector: Regex
├── post/             # FuzzyPostProcessor
├── types.py          # NEW: Intent, Language, PredictionResult, TravelEntity
├── interfaces.py     # ABCs
├── pipeline.py       # REFACTORED: composes modular components
├── fuzzy_matcher.py
└── preprocessor.py
```

**Design Rationale:**
| Task | Regex | SpaCy | CamemBERT | Why |
|------|-------|-------|-----------|-----|
| Entity | ✓ | ✓ | ✓ | All valid NER approaches |
| Intent | ✓ | ✗ | ✓ | SpaCy not suited for classification |
| Language | ✓ | ✗ | ✗ | Simple patterns, ML overkill |

---

## [0.3.0] Modular Evaluation & Repository Cleanup - 2025-01-22

### Added
- **Modular Evaluation Framework** (`src/evaluation/`):
  - `cli.py` - Main CLI entry point (`python -m src.evaluation`)
  - `data_loader.py` - Dataset loading and normalization
  - `metrics.py` - Result dataclasses (IntentResults, EntityResults, CombinedResults, etc.)
  - `reporting.py` - Table printing and JSON export
  - `progress.py` - Progress callback protocol for batch processing
  - `evaluators/` - Modular evaluators (intent, entity, language, combined)
- **Progress Callbacks** for batch processing:
  - `CamembertIntentClassifier.classify_batch()` - progress_callback parameter
  - `CamembertEntityExtractor.extract_batch()` - progress_callback parameter
  - Progress shows after each batch (default 128 samples) instead of percentage

### Changed
- **src/main.py** - Fixed broken extractor imports:
  - Updated EXTRACTORS dict to use new modular paths (`src.nlp.entity.*`)
  - Changed `extract_entities()` to `extract()` (new interface)
  - Removed deprecated "fuzzy" option (use SpaCy + FuzzyPostProcessor manually)
- **Makefile** - Updated targets:
  - `evaluate` / `evaluate-full` now use `python -m src.evaluation`
  - Removed broken targets: `train-baseline`, `train-camembert`, `train-flan`, `run-api`, `demo-fuzzy`
- **.gitignore** - Added `trajet_*.html` pattern for generated map files

### Removed
- **Duplicate Code** (~1,150 lines):
  - `src/nlp/entity_extractor.py` (721 lines) - replaced by modular `src/nlp/entity/`
  - `src/nlp/models/{ensemble,flan_t5,spacy,camembert}_model.py` - 4 empty files
  - `src/nlp/entity/{flant5,mistral}_entity.py` - stub files
  - `src/nlp/intent/{flant5,mistral}_intent.py` - stub files
- **Empty Files/Directories**:
  - `training/*.py` - 5 empty training scripts (kept directory with .gitkeep)
  - `src/api/` - entire empty API module
  - `notebooks/*.ipynb` - 5 empty notebooks (kept directory with .gitkeep)
- **Orphaned Files**:
  - `wandb.py` - generic W&B template
  - `trajet_AUTO.html` - generated output file
- **Old Evaluation Module**:
  - `evaluation/` directory (replaced by `src/evaluation/`)

### Architecture
```
src/evaluation/           # NEW modular evaluation
├── cli.py               # Main entry point
├── data_loader.py       # Dataset loading
├── metrics.py           # Result dataclasses
├── reporting.py         # Table printing
├── progress.py          # Progress callbacks
└── evaluators/          # Modular evaluators
    ├── intent.py
    ├── entity.py
    ├── language.py
    └── combined.py

src/nlp/entity/          # KEPT (3 extractors)
├── regex_entity.py
├── spacy_entity.py
└── camembert_entity.py

src/nlp/intent/          # KEPT (2 classifiers)
├── regex_intent.py
└── camembert_intent.py
```

## [0.2.2] Dataset Architecture Refactoring - 2025-01-22

### Added
- **Separate Dataset Generation & Augmentation** (academic rigor):
  - `datasets/scripts/generate_base.py` - Generates clean dataset without STT errors
  - `datasets/scripts/augment_stt.py` - Applies STT errors to base dataset
- **Two-Level Dataset Architecture**:
  - `datasets/base/` - Clean data (no STT errors), IDs prefixed `BASE`
  - `datasets/augmented/` - With STT errors applied, IDs prefixed `STT`
- **TRIP-Safe Augmentation**: Disabled `incomplete` error type for TRIP entries to preserve ground truth station names

### Changed
- **Directory Rename**: `datasets/generated/` → `datasets/base/`
- **File Naming**: `stt_dataset_100k.csv` → `base_dataset_100k.csv` (in base/)
- **ID Format**: `STT000001` → `BASE000001` (in base dataset)
- **Documentation**: Updated `datasets/base/DATASET.md` for new architecture

### Removed
- **Obsolete Files**:
  - `datasets/scripts/augment_data.py` (empty file)
  - `datasets/scripts/generate_sentences.py` (replaced by generate_base.py)
  - `datasets/scripts/split_dataset.py` (integrated in generate_base.py)
  - `datasets/scripts/generate_camembert_data.py` (old generator)
  - `datasets/scripts/preprocess_camembert_data.py` (old preprocessor)
  - `datasets/processed/` (empty directory)

### Benefits
- **Ablation Studies**: Compare model performance on clean vs STT-augmented data
- **Reproducibility**: Same base data can be augmented multiple times with different seeds
- **Academic Rigor**: Clear separation between data generation and noise injection

## [0.2.1] 100k STT Dataset with Intermediate Stops - 2025-01-22

### Added
- **STT Augmentation Module** (`src/data/stt_augmentation.py`):
  - `STTErrorConfig` - Configurable error rates for 14 STT error types
  - `STTAugmenter` - Apply realistic Whisper transcription errors to text
  - `create_augmenter(intensity)` - Factory with preset profiles (clean, light, moderate, heavy)
  - 14 error types: filler words, false starts, repetitions, phonetic confusion, station misspelling, punctuation/capitalization errors, code-switching, hallucinations, noise artifacts, homophones
- **100k STT Dataset Generator** (`datasets/scripts/generate_stt_dataset.py`):
  - 200+ templates per category (TRIP, NOT_TRIP, UNKNOWN)
  - Multi-language support: FRENCH (76%), ENGLISH (10%), SPANISH (3.2%), GERMAN (3.2%), ITALIAN (1.5%), UNKNOWN (6%)
  - Intent and language as separate columns (cleaner design)
  - **Intermediate stops support**: 15% of TRIP entries include "via" routes
  - Stratified train/val/test split (70/15/15)
  - Reproducible generation with seed parameter
- **Intermediate Stop Templates**:
  - French: 30 templates ("en passant par", "via", "puis...puis")
  - English: 10 templates ("via", "through", "stopping at")
  - Spanish: 5 templates ("pasando por", "con parada en")
  - German: 5 templates ("über", "mit Halt in")
  - Italian: 5 templates ("passando per", "con fermata a")
- **Generated Dataset** (`datasets/generated/`):
  - `stt_dataset_100k.csv` / `.json` - Full 100k dataset
  - `train.csv` / `.json` - 70k training samples
  - `val.csv` / `.json` - 15k validation samples
  - `test.csv` / `.json` - 15k test samples
- **Dataset Documentation** (`datasets/generated/DATASET.md`):
  - French documentation explaining dataset structure and distribution
  - Examples for all intent/language combinations
  - STT error types reference table

### Changed
- **DatasetEntry schema**: Added `intermediate` field for via stations
- **Export functions**: CSV/JSON now include 7 columns (added `intermediate`)
- **Statistics output**: Now shows intermediate stops distribution

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total entries | 100,000 |
| TRIP intent | 70,000 (70%) |
| NOT_TRIP intent | 25,000 (25%) |
| UNKNOWN intent | 5,000 (5%) |
| TRIP with intermediate | 10,348 (14.8% of TRIP) |
| Languages | 6 (FR, EN, ES, DE, IT, UNKNOWN) |

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
