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
│                            NLP PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐     ┌──────────────────┐     ┌────────────────────┐     │
│  │  Preprocessor  │────▶│ Intent Classifier│────▶│ Entity Extractor   │     │
│  │  (normalisation,│     │ (voyage/non-voyage)│   │ (DEP, DEST, INTER) │     │
│  │  tokenization) │     └──────────────────┘     └─────────┬──────────┘     │
│  └────────────────┘                                        │                │
│                                                            ▼                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         MODELES NLP                                   │   │
│  ├──────────┬──────────┬──────────────┬──────────────┬─────────────────┤   │
│  │ Baseline │  SpaCy   │  CamemBERT   │   Flan-T5    │    Ensemble     │   │
│  │  Regex   │   NER    │  (fine-tuned)│   (seq2seq)  │   (voting)      │   │
│  └──────────┴──────────┴──────────────┴──────────────┴─────────────────┘   │
│                                         │                                   │
│                                   ┌─────▼──────┐                           │
│                                   │  Station   │                           │
│                                   │  Matcher   │                           │
│                                   │  (fuzzy)   │                           │
│                                   └─────┬──────┘                           │
└─────────────────────────────────────────┼───────────────────────────────────┘
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

### API REST

```bash
# Demarrer le serveur
python -m src.api.app

# Requete
curl -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Je veux aller de Paris a Lyon"}'
```

---

## Benchmarks

> Evalue sur `datasets/splits/short-splits/test.csv` (1,500 samples)
> Architecture modulaire: Intent classifiers + Entity extractors + Post-processors (Fuzzy)

### Table 1: Intent Classification

| Modele | Accuracy | Latence |
|--------|----------|---------|
| Regex | 71.9% | 0.07ms |
| CamemBERT | 34.9% | 26.6ms |

### Table 2: Entity Extraction (sans fuzzy)

| Modele | Accuracy | Precision | Recall | F1-Score | Latence |
|--------|----------|-----------|--------|----------|---------|
| Regex | 6.0% | 10.6% | 9.7% | 10.1% | 0.03ms |
| SpaCy | 11.0% | 19.7% | 11.8% | 14.5% | 2.7ms |
| CamemBERT | 12.7% | 30.8% | 22.0% | 25.3% | 1.3ms |

### Table 3: Entity Extraction + Fuzzy Post-Processing

| Modele | Accuracy | Precision | Recall | F1-Score | Latence |
|--------|----------|-----------|--------|----------|---------|
| Regex + Fuzzy | 22.3% | 41.5% | 37.7% | 39.4% | 32.4ms |
| SpaCy + Fuzzy | 20.4% | 36.5% | 22.4% | 27.3% | 1.5ms |
| CamemBERT + Fuzzy | 28.6% | 49.7% | 35.2% | 40.6% | 3.7ms |

### Table 4: Combined Pipeline (Intent + Entity)

| Intent | Entity | Intent Acc | Entity Acc | Latence |
|--------|--------|------------|------------|---------|
| Regex | Regex | 71.9% | 6.0% | 0.1ms |
| Regex | SpaCy | 71.9% | 11.0% | 6.7ms |
| Regex | CamemBERT | 71.9% | 12.7% | 1.3ms |
| CamemBERT | Regex | 34.9% | 6.0% | 26.9ms |
| CamemBERT | SpaCy | 34.9% | 11.0% | 35.1ms |
| CamemBERT | CamemBERT | 34.9% | 12.7% | 27.8ms |

### Table 5: Combined Pipeline + Fuzzy

| Intent | Entity | Intent Acc | Entity Acc | Latence |
|--------|--------|------------|------------|---------|
| **Regex** | **Regex + Fuzzy** | **71.9%** | **22.3%** | **1.4ms** |
| Regex | SpaCy + Fuzzy | 71.9% | 20.4% | 7.1ms |
| **Regex** | **CamemBERT + Fuzzy** | **71.9%** | **28.6%** | **1.4ms** |
| CamemBERT | Regex + Fuzzy | 34.9% | 22.3% | 26.4ms |
| CamemBERT | SpaCy + Fuzzy | 34.9% | 20.4% | 34.6ms |
| CamemBERT | CamemBERT + Fuzzy | 34.9% | 28.6% | 28.0ms |

**Best Pipeline**: Regex (intent) + CamemBERT (entity) + Fuzzy (post-processing)

*Script: `poetry run python evaluation/evaluate_all.py --eval-type all --dataset datasets/splits/short-splits/test.csv --output-json evaluation_results.json`*

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

| Nom | Role | Contact |
|-----|------|---------|
| Romain Bernier | | email@epitech.eu |
| Victor Vattier | | email@epitech.eu |
| Marine Gayet | | email@epitech.eu |
| Camille Kerserho | | email@epitech.eu |

---

## License

MIT License - voir [LICENSE](LICENSE)

---

<p align="center">
  <i>Projet Epitech T9-AIA - Travel Order Resolver</i>
</p>
