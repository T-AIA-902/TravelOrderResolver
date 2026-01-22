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
│                     NLP PIPELINE (src/nlp/)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  interfaces.py: LanguageDetector | IntentClassifier | EntityExtractor |     │
│                 PostProcessor (ABCs)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
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

> Evalue sur `datasets/augmented/test.csv` (15,000 samples avec erreurs STT simulees)
> Architecture modulaire: Language detectors + Intent classifiers + Entity extractors + Post-processors

### Table 1: Language Detection

| Modele | Overall | FR | EN | UNK | Latence |
|--------|---------|----|----|-----|---------|
| Regex | 68.9% | 68% | 68% | 76% | 0.03ms |
| Langdetect | 73.8% | 78% | 65% | 58% | 9.69ms |

*Note: ES, DE, IT mappes vers UNKNOWN (langues non supportees). Etude focalisee FR/EN.*

### Table 2: Intent Classification (per-language)

| Modele | Overall | FR | EN | UNK | Latence |
|--------|---------|----|----|-----|---------|
| Regex | 65.5% | 67% | 73% | 53% | 0.01ms |
| CamemBERT | 70.0% | 74% | 68% | 51% | 27.52ms |

### Table 3: Entity Extraction

| Modele | Fuzzy | Accuracy | Precision | Recall | F1 | Latence |
|--------|-------|----------|-----------|--------|-----|---------|
| Regex | - | 32.9% | 47.1% | 43.3% | 45.1% | 0.0ms |
| Regex | ✓ | 56.8% | 76.5% | 70.6% | 73.4% | 16.8ms |
| SpaCy | - | 25.1% | 57.7% | 35.5% | 43% | 1.9ms |
| SpaCy | ✓ | 31.4% | 69.9% | 42.8% | 52% | 1.5ms |
| CamemBERT | - | 15.1% | 45.3% | 29.5% | 35% | 1.2ms |
| CamemBERT | ✓ | 30.4% | 67.8% | 44.0% | 52% | 1.2ms |

### Table 4: Combined Pipeline

| Intent | Entity | Fuzzy | Intent Acc | Entity Acc | Latence |
|--------|--------|-------|------------|------------|---------|
| Regex | Regex | ✓ | 65.5% | 56.8% | 0.1ms |
| Regex | SpaCy | ✓ | 65.5% | 31.4% | 6.7ms |
| Regex | CamemBERT | ✓ | 65.5% | 30.4% | 1.2ms |
| CamemBERT | Regex | ✓ | 70.0% | 56.8% | 27.4ms |
| CamemBERT | SpaCy | ✓ | 70.0% | 31.4% | 35.6ms |
| CamemBERT | CamemBERT | ✓ | 70.0% | 30.4% | 28.7ms |

**Best configurations:**
- **Speed-optimized:** Regex + Regex + Fuzzy (0.1ms, 56.8% entity accuracy)
- **Quality-optimized:** CamemBERT + Regex + Fuzzy (27.4ms, 70.0% intent, 56.8% entity)

### Table 5: Ablation Study (Clean vs STT)

| Dataset | Intent Acc | Entity Acc | Language Acc |
|---------|------------|------------|--------------|
| Clean (base/) | TBD | TBD | TBD |
| STT-augmented (augmented/) | TBD | TBD | TBD |
| Delta | TBD | TBD | TBD |

*Pipeline de reference pour comparaison*

*Script: `poetry run python -m src.evaluation --eval-type all --dataset datasets/augmented/test.csv`*

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
| Victor Vattier | Référent ML & Dev | [@VictorVattierEpitech](https://github.com/VictorVattierEpitech) | victor.vattier@epitech.eu |
| Marine Gayet | Référente Frontend & Dev | [@Marinegyt](https://github.com/Marinegyt) | marine.gayet@epitech.eu |
| Camille Kerserho | Dev | [@Camserho](https://github.com/Camserho) | camille.kerserho@epitech.eu |

---

## License

MIT License - voir [LICENSE](LICENSE)

---

<p align="center">
  <i>Projet Epitech T9-AIA - Travel Order Resolver</i>
</p>
