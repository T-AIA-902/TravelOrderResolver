# TravelOrderResolver

> NLP-based travel order parsing for SNCF railway network (Epitech T9-AIA)

## Project Overview
- **Goal**: Parse French natural language travel requests → extract stations → compute optimal routes
- **Pipeline**: `Raw Text → PRE → Language → Intent → Entity → POST → Pathfinding`
- **Python**: 3.10+

## Always Read First
- @README.md - Project overview & architecture diagram
- @CHANGELOG.md - Latest changes (check before modifying)
- @BACKLOG.md - Task backlog with priorities
- @.claude/REPOSITORY.md - Full repo structure
- @pyproject.toml - Dependencies

## Architecture (src/nlp/)
```
src/nlp/
├── interfaces.py    # ABCs: IntentClassifier, EntityExtractor, PostProcessor
├── types.py         # Intent, Language, TravelEntity, PredictionResult
├── pre/             # STTArtifactFilter, Preprocessor
├── language/        # RegexLanguageDetector, LangdetectLanguageDetector
├── intent/          # RegexIntentClassifier, SpacyIntentClassifier, CamembertIntentClassifier
├── entity/          # RegexEntityExtractor, SpacyEntityExtractor, CamembertEntityExtractor
└── post/            # FuzzyPostProcessor, StationMatcher
```

**When adding new models**: Inherit from ABCs in `interfaces.py`

## DRY Principles
- Scan the codebase to verify a similar function doesn't exist
- If a similar function exists but cannot be reused, it must be refactored
- **When adding new models**: Inherit from ABCs in `src/nlp/interfaces.py`
- Reuse existing components:
  - Pre-processing: `src/nlp/pre/`
  - Fuzzy matching: `src/nlp/post/station_matcher.py`
  - Metrics: `src/evaluation/metrics.py`
  - Device utils: `src/utils/device.py`

## Testing & Evaluation
```bash
make lint          # flake8, mypy, isort, black
make format        # Auto-format code
make test          # Run all tests with coverage
make test-unit     # Unit tests only
```

**At the end of every plan**:
1. Run `make lint && make test-unit` to verify changes
2. Update `CHANGELOG.md` with completed features
3. Tick completed tasks in `BACKLOG.md`
4. If files/folders were added or removed, update `.claude/REPOSITORY.md`

## Code Style
- Type hints on all function signatures
- Follow existing patterns in the module you're modifying
- No single-letter variables except i, j, k in loops
- Use pathlib.Path for file operations

## Branch Workflow
- Create feature branches from `dev`: `git checkout -b feat/feature-name`
- **NEVER commit without explicit user request** - user will stage files and commit manually
