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

### Frontend (Docker)

```bash
# Build et lancer le frontend sur http://localhost:3000
docker build -f docker/Dockerfile.frontend -t tor-frontend .
docker run -d --name tor-frontend -p 3000:80 tor-frontend
```

Voir [src/frontend/README.md](src/frontend/README.md) pour le détail (dev local, structure, API attendue).

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

| Pipeline Stage | Best Model | Accuracy | Latency |
|----------------|------------|----------|---------|
| Pre-Processing | STT Filter + Normalizer | - | <1ms |
| Language Detection | Langdetect | 77.9% | 6.1ms |
| Intent Classification | SpaCy | 79.1% | 0.8ms |
| Entity Extraction | Regex + Fuzzy | 57.1% | 9.8ms |

**Recommended Configuration:** SpaCy (Intent) + Regex (Entity) + Fuzzy (Post-Processing)

For detailed analysis, see:
- [BENCHMARK.md](docs/BENCHMARK.md) - Full evaluation tables (pre-processing impact, component comparison, language analysis)
- [full_evaluation.ipynb](notebooks/full_evaluation.ipynb) - Comprehensive benchmark of all models
- [single_evaluation.ipynb](notebooks/single_evaluation.ipynb) - Quick benchmark for testing your model

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
