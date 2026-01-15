# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- Test discovery issue: added missing `tests/unit/__init__.py`

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
