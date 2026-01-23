TravelOrderResolver/
├── .claude/
│   ├── CLAUDE.md                     # Project rules (auto-loaded)
│   ├── REPOSITORY.md                 # Full repo structure
│   ├── settings.json                 # Permissions config
│   └── rules/
│       └── nlp.md                    # Path-specific rules for src/nlp/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── task.md
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── model-benchmark.yml
│   │   └── release.yml
│   └── PULL_REQUEST_TEMPLATE.md
│
├── datasets/
│   ├── augmented/                    # With STT errors applied
│   │   ├── stt_dataset_100k.csv/.json
│   │   ├── train.csv/.json
│   │   ├── val.csv/.json
│   │   └── test.csv/.json
│   ├── base/                         # Clean data (no STT errors)
│   │   ├── base_dataset_100k.csv/.json
│   │   ├── train.csv/.json
│   │   ├── val.csv/.json
│   │   └── test.csv/.json
│   ├── raw/
│   │   ├── audio/
│   │   ├── sentences/
│   │   └── sncf/
│   │       ├── gares-de-voyageurs.json
│   │       ├── lignes-par-type.json
│   │       ├── liste-des-gares.json
│   │       └── tgvmax.json
│   ├── scripts/
│   │   ├── augment_stt.py
│   │   ├── generate_base.py
│   │   ├── import_sncf_data.py
│   │   └── validate_dataset.py
│   └── README.md
│
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.training
│   └── docker-compose.yml
│
├── docs/
│   ├── api/
│   ├── architecture/
│   │   └── diagrams/
│   ├── experiments/
│   ├── rapport/
│   │   └── src/
│   └── BENCHMARK.md
│
├── models/
│   ├── camembert-ner/                # Pre-trained CamemBERT tokenizer
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── special_tokens_map.json
│   │   └── added_tokens.json
│   ├── checkpoints/
│   ├── experiments/
│   └── production/
│
├── notebooks/
│   └── evaluation.ipynb
│
├── scripts/
│   ├── download_models.sh
│   ├── generate_report.sh
│   ├── run_benchmark.sh
│   └── setup_dev.sh
│
├── src/
│   ├── __init__.py
│   ├── main.py                       # CLI entry point
│   │
│   ├── data/                         # Data management
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── schedule_database.py
│   │   ├── station_database.py       # 6,101 SNCF stations
│   │   └── stt_augmentation.py       # STT error simulation
│   │
│   ├── evaluation/                   # Evaluation module
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── cli.py                    # CLI: python -m src.evaluation
│   │   ├── confusion.py
│   │   ├── data_loader.py
│   │   ├── formatting.py
│   │   ├── metrics.py                # Result dataclasses
│   │   ├── model_factory.py
│   │   ├── preprocessing.py
│   │   ├── progress.py               # Progress callbacks
│   │   ├── reporting.py              # Table printing, JSON export
│   │   ├── evaluators/
│   │   │   ├── __init__.py
│   │   │   ├── combined.py
│   │   │   ├── entity.py
│   │   │   ├── intent.py
│   │   │   └── language.py
│   │   └── reports/                  # Generated JSON reports
│   │
│   ├── monitoring/                   # Monitoring (bonus)
│   │   ├── __init__.py
│   │   ├── carbon_calculator.py
│   │   ├── metrics_logger.py
│   │   └── resource_tracker.py
│   │
│   ├── nlp/                          # NLP module (core)
│   │   ├── __init__.py
│   │   ├── interfaces.py             # ABCs: IntentClassifier, EntityExtractor, PostProcessor
│   │   ├── pipeline.py               # Main NLP pipeline
│   │   ├── types.py                  # Intent, Language, TravelEntity, PredictionResult
│   │   ├── entity/                   # Entity extraction (NER)
│   │   │   ├── __init__.py
│   │   │   ├── camembert_entity.py
│   │   │   ├── regex_entity.py
│   │   │   └── spacy_entity.py
│   │   ├── intent/                   # Intent classification
│   │   │   ├── __init__.py
│   │   │   ├── camembert_intent.py
│   │   │   ├── regex_intent.py
│   │   │   └── spacy_intent.py
│   │   ├── language/                 # Language detection
│   │   │   ├── __init__.py
│   │   │   ├── langdetect_language.py
│   │   │   └── regex_language.py
│   │   ├── post/                     # Post-processing
│   │   │   ├── __init__.py
│   │   │   ├── fuzzy_post_processor.py
│   │   │   └── station_matcher.py
│   │   ├── pre/                      # Pre-processing
│   │   │   ├── __init__.py
│   │   │   ├── preprocessor.py       # Text normalization
│   │   │   └── stt_filter.py         # STT artifact cleaning
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── hf_batching.py        # HuggingFace GPU batching
│   │
│   ├── pathfinding/                  # Pathfinding module
│   │   ├── __init__.py
│   │   ├── graph.py                  # NetworkX railway graph
│   │   ├── route_optimizer.py
│   │   ├── schedule_manager.py
│   │   └── algorithms/
│   │       ├── __init__.py
│   │       ├── astar.py
│   │       ├── bellman_ford.py
│   │       └── dijkstra.py
│   │
│   ├── speech/                       # Speech-to-Text (bonus)
│   │   ├── __init__.py
│   │   ├── audio_processor.py
│   │   ├── transcriber.py
│   │   └── whisper_model.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── device.py                 # GPU/CPU detection
│   │   ├── logger.py
│   │   └── validators.py
│   │
│   └── visualization/                # Map visualization
│       ├── __init__.py
│       └── map_visualizer.py         # Folium interactive maps
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── e2e/
│   ├── integration/
│   └── unit/
│       ├── __init__.py
│       ├── test_data_parsing.py
│       ├── test_entity_extractor.py
│       ├── test_fuzzy_matcher.py
│       ├── test_intent_classifier.py
│       └── test_nlp.py
│
├── training/
│   └── __init__.py                   # (scripts to be added)
│
├── .env.example
├── .flake8
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── BACKLOG.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── poetry.lock
├── pyproject.toml
└── README.md
