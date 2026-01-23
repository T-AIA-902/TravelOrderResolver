# Travel Order Resolver

[![CI](https://github.com/Romain-Ber/TravelOrderResolver/actions/workflows/ci.yml/badge.svg)](https://github.com/Romain-Ber/TravelOrderResolver/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-0%25-red)](https://github.com/Romain-Ber/TravelOrderResolver)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Systeme intelligent de traitement de commandes de voyage en langage naturel pour le reseau ferroviaire SNCF.

**Projet Epitech T9-AIA** | NLP + Pathfinding + Speech-to-Text

---

## Table des matieres

- [Apercu](#apercu)
- [Fonctionnalites](#fonctionnalites)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Benchmarks](#benchmarks)
- [Documentation](#documentation)
- [Contribution](#contribution)
- [Equipe](#equipe)

---

## Apercu

Travel Order Resolver est un programme NLP capable de:

1. **Analyser** des phrases en francais exprimant une demande de voyage
2. **Extraire** les gares de depart et d'arrivee (+ arrets intermediaires)
3. **Calculer** l'itineraire optimal via le reseau SNCF
4. **Transcrire** des commandes vocales (bonus)

### Exemple

```
Input:  "Je voudrais aller de Paris a Lyon en passant par Dijon"
Output: 1,Paris,Dijon,Lyon

Input:  "Quel temps fait-il demain ?"
Output: 1,NOT_TRIP
```

---

## Fonctionnalites

### Core
- [ ] Classification d'intention (voyage vs non-voyage)
- [ ] Extraction NER (depart, destination, intermediaires)
- [ ] Matching fuzzy des noms de gares
- [ ] Detection de langue (francais requis)
- [ ] Gestion des fautes d'orthographe
- [ ] Pathfinding (Dijkstra/A*)

### Bonus
- [ ] Speech-to-Text (Whisper offline)
- [ ] Arrets intermediaires
- [ ] Benchmarking multi-modeles
- [ ] Monitoring CPU/RAM/Carbone
- [ ] API REST
- [ ] Interface web demo (Gradio)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                     │
├─────────────┬───────────────────────────────────────┬───────────────────────┤
│    Audio    │              Text                     │       API REST        │
│    (WAV)    │        (stdin/file/URL)               │      (FastAPI)        │
└──────┬──────┴──────────────────┬────────────────────┴───────────┬───────────┘
       │                         │                                │
       ▼                         │                                │
┌──────────────┐                 │                                │
│ Speech-to-   │                 │                                │
│ Text (Whisper)│                │                                │
└──────┬───────┘                 │                                │
       │                         │                                │
       └─────────────────────────┼────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NLP PIPELINE (src/nlp/)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  interfaces.py: LanguageDetector | IntentClassifier | EntityExtractor |     │
│                 PostProcessor (ABCs)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    PRE-PROCESSING (src/nlp/pre/)                     │  │
│  │     STTArtifactFilter: [noise], [music], [inaudible], etc.           │  │
│  │     Preprocessor: unicode, tokenization, accents, hyphens            │  │
│  └──────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐    │
│  │ LANGUAGE DETECTOR│  │  INTENT CLASSIFIER  │  │  ENTITY EXTRACTOR   │    │
│  │   (language/)    │  │     (intent/)       │  │     (entity/)       │    │
│  └────────┬─────────┘  └──────────┬──────────┘  └──────────┬──────────┘    │
│           │                       │                        │               │
│           └───────────────────────┼────────────────────────┘               │
│                                   │                                        │
│                                   ▼                                        │
│                        ┌────────────────┐                                  │
│                        │ POST-PROCESSOR │                                  │
│                        │    (post/)     │                                  │
│                        └───────┬────────┘                                  │
│                                │                                           │
└────────────────────────────────┼───────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PATHFINDING ENGINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐    ┌─────────────────┐    ┌────────────────────────────┐    │
│  │   Graph    │───▶│   Algorithmes   │───▶│     Route Optimizer        │    │
│  │  Builder   │    │  Dijkstra / A*  │    │  (intermediaires, attentes)│    │
│  │  (SNCF)    │    └─────────────────┘    └────────────────────────────┘    │
│  └────────────┘                                                             │
│                     ┌─────────────────────────────────────────────┐         │
│                     │  Multi-transports: TGV | TER | Intercites   │         │
│                     └─────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
┌──────────────────┐         ┌─────────────────────┐         ┌────────────────┐
│   CLI Output     │         │     API Response    │         │   Web Demo     │
│ ID,Dep,Step,Dest │         │       (JSON)        │         │   (Gradio)     │
└──────────────────┘         └─────────────────────┘         └────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         MONITORING & OBSERVABILITE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │  CPU/RAM    │    │   Latence   │    │  Empreinte  │    │   Metriques  │  │
│  │  Tracking   │    │   Requetes  │    │   Carbone   │    │   ML (F1..)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequis
- Python 3.10+
- Poetry (recommande) ou pip
- GPU CUDA (optionnel, pour entrainement)

### Installation rapide

```bash
# Cloner le repository
git clone https://github.com/Romain-Ber/TravelOrderResolver.git
cd TravelOrderResolver

# Installer les dependances
make install

# Telecharger les modeles
make download-models

# Verifier l'installation
make test
```

### Installation manuelle

```bash
# Avec Poetry
poetry install
poetry run python -m spacy download fr_dep_news_trf

# Avec pip
pip install -r requirements.txt
python -m spacy download fr_dep_news_trf
```

---

## Utilisation

### CLI (Mode principal)

```bash
# Depuis stdin
echo "1,Je veux aller de Paris a Lyon" | python -m src.main

# Depuis un fichier
python -m src.main --input sentences.csv --output results.csv

# Depuis une URL
python -m src.main --input https://example.com/sentences.csv

# Mode interactif
python -m src.main --interactive
```

### NLP uniquement (pour evaluation)

```bash
# Module NLP isole
python -m src.nlp.pipeline --input sentences.csv
```

### Speech-to-Text

```bash
# Transcription audio
python -m src.speech.transcriber --audio recording.wav

# Pipeline complet (audio -> itineraire)
python -m src.main --audio recording.wav
```

### API REST (Not Yet Implemented)

```bash
# API REST not yet available
# See TASKS.md section 6.2 for planned endpoints
```

---

## Benchmarks

> Evalue sur `datasets/augmented/test.csv` (15,005 samples avec erreurs STT simulees)
> Format: sklearn `classification_report` style

### Language Detection

```
========================================================================
LANGUAGE DETECTION: Regex                              Accuracy: 67.5%
========================================================================
                Precision     Recall         F1    Support

fr                   0.98       0.68       0.80      13366
en                   0.70       0.66       0.68        966
unk                  0.09       0.68       0.16        673

macro avg            0.59       0.67       0.55      15005
latency                                              0.0ms
========================================================================

========================================================================
LANGUAGE DETECTION: Langdetect                         Accuracy: 77.9%
========================================================================
                Precision     Recall         F1    Support

fr                   0.96       0.80       0.87      13366
en                   0.48       0.66       0.55        966
unk                  0.14       0.55       0.23        673

macro avg            0.53       0.67       0.55      15005
latency                                              5.4ms
========================================================================
```

### Intent Classification

```
========================================================================
INTENT CLASSIFICATION: Regex                           Accuracy: 65.8%
========================================================================
                Precision     Recall         F1    Support

TRIP                 0.76       0.80       0.78      11101
NOT_TRIP             0.29       0.25       0.27       3754
UNKNOWN              0.75       0.42       0.54        150

macro avg            0.60       0.49       0.53      15005
latency                                              0.0ms
========================================================================

========================================================================
INTENT CLASSIFICATION: SpaCy                           Accuracy: 79.1%
========================================================================
                Precision     Recall         F1    Support

TRIP                 0.94       0.78       0.85      11101
NOT_TRIP             0.56       0.85       0.67       3754
UNKNOWN              0.00       0.00       0.00        150

macro avg            0.50       0.54       0.51      15005
latency                                              1.0ms
========================================================================

========================================================================
INTENT CLASSIFICATION: CamemBERT                       Accuracy: 64.2%
========================================================================
                Precision     Recall         F1    Support

TRIP                 0.72       0.85       0.78      11101
NOT_TRIP             0.06       0.03       0.04       3754
UNKNOWN              0.75       0.42       0.54        150

macro avg            0.51       0.43       0.45      15005
latency                                              4.0ms
========================================================================
```

*Note: CamemBERT uses `almanach/camembert-base` (not fine-tuned for intent classification).*
*SpaCy achieves best accuracy (79.1%) but cannot detect UNKNOWN class.*

### Entity Extraction

```
========================================================================
ENTITY EXTRACTION: Regex                               Accuracy: 33.3%
========================================================================
                Precision     Recall         F1    Support

departure            0.45       0.44       0.45       9834
destination          0.49       0.43       0.46      11000

macro avg            0.47       0.44       0.45      20834
latency                                              0.0ms
========================================================================

========================================================================
ENTITY EXTRACTION: Regex + Fuzzy                       Accuracy: 57.1%
========================================================================
                Precision     Recall         F1    Support

departure            0.77       0.75       0.76       9834
destination          0.76       0.67       0.71      11000

macro avg            0.76       0.71       0.73      20834
latency                                              9.2ms
========================================================================

========================================================================
ENTITY EXTRACTION: SpaCy                               Accuracy: 27.8%
========================================================================
                Precision     Recall         F1    Support

departure            0.64       0.32       0.43       9834
destination          0.53       0.42       0.47      11000

macro avg            0.58       0.37       0.45      20834
latency                                              1.1ms
========================================================================

========================================================================
ENTITY EXTRACTION: SpaCy + Fuzzy                       Accuracy: 34.2%
========================================================================
                Precision     Recall         F1    Support

departure            0.76       0.39       0.51       9834
destination          0.64       0.50       0.56      11000

macro avg            0.70       0.44       0.54      20834
latency                                              2.0ms
========================================================================
```

### Recommended Configuration

| Use Case | Intent | Entity | Fuzzy | Intent Acc | Entity Acc | Latency |
|----------|--------|--------|-------|------------|------------|---------|
| **Best Accuracy** | SpaCy | Regex | ✓ | 79.1% | 57.1% | ~10ms |
| **Best Balance** | Regex | Regex | ✓ | 65.8% | 57.1% | ~9ms |
| **Lowest Latency** | Regex | Regex | - | 65.8% | 33.3% | <1ms |

*Run: `poetry run python -m src.evaluation --eval-type all`*

---

## Documentation

- [Architecture detaillee](docs/architecture/system_design.md)
- [Pipeline NLP](docs/architecture/nlp_pipeline.md)
- [Guide d'entrainement](docs/experiments/experiment_log.md)
- [Rapport technique (PDF)](docs/rapport/rapport_technique.pdf)
- [API Reference](docs/api/)

---

## Contribution

Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines.

```bash
# Setup dev
make dev-setup

# Lancer les tests
make test

# Linter + formatter
make lint

# Pre-commit hooks
pre-commit install
```

---

## Equipe

| Nom | Role | Github | Contact Epitech |
|-----|------|--------|-----------------|
| Romain Bernier | Architecte | [@Romain-Ber](https://github.com/Romain-Ber) | romain.bernier@epitech.eu |
| Victor Vattier | Référent ML | [@VictorVattierEpitech](https://github.com/VictorVattierEpitech) | victor.vattier@epitech.eu |
| Marine Gayet | Dev Backend & ML | [@Marinegyt](https://github.com/Marinegyt) | marine.gayet@epitech.eu |
| Camille Kerserho | Dev Backend & ML | [@Camserho](https://github.com/Camserho) | camille.kerserho@epitech.eu |

---

## License

MIT License - voir [LICENSE](LICENSE)

---

<p align="center">
  <i>Projet Epitech T9-AIA - Travel Order Resolver</i>
</p>
