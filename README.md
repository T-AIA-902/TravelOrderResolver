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
- [x] Speech-to-Text (Whisper offline)
- [ ] Arrets intermediaires
- [ ] Benchmarking multi-modeles
- [x] Monitoring CPU/RAM/Carbone
- [x] Monitoring infrastructure (Prometheus, Grafana, Gatus)
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
- Docker + Docker Compose (optionnel, pour monitoring infrastructure)

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

### Dependances optionnelles (ML)

```bash
# Speech-to-Text (Whisper) + Carbon tracking + Experiment tracking
poetry install --with ml

# Ou avec pip
pip install openai-whisper sounddevice soundfile codecarbon mlflow
```

---

## Utilisation

### CLI (Mode principal)

```bash
# Mode interactif (texte, extracteur CamemBERT par defaut)
python -m src.main --extractor camembert

# Mode interactif avec speech-to-text (Windows PowerShell uniquement)
python -m src.main --extractor camembert --speech

# Autres extracteurs disponibles : regex, spacy
python -m src.main --extractor regex
python -m src.main --extractor spacy
```

> **Note :** Le mode `--speech` necessite Windows PowerShell (le micro n'est pas accessible en WSL).
> Appuyez sur Entree sans texte pour enregistrer depuis le micro, ou tapez votre demande directement.

### NLP uniquement (pour evaluation)

```bash
# Module NLP isole
python -m src.nlp.pipeline --input sentences.csv
```

### Speech-to-Text (Whisper)

```python
from src.speech import SpeechTranscriber

transcriber = SpeechTranscriber(model_name="medium")

# Enregistrement micro avec duree fixe (5 secondes)
result = transcriber.transcribe_from_mic(duration=5.0)
print(result.text)      # "Je veux aller de Paris a Lyon"
print(result.language)   # "fr"

# Enregistrement micro avec detection de silence
result = transcriber.transcribe_from_mic_auto()
print(result.text)
```

```python
from src.speech import WhisperModel

# Transcription d'un fichier audio
model = WhisperModel(model_name="medium")
result = model.transcribe("recording.wav")
print(result.text)
```

### Monitoring

#### Tracking Python (CPU/RAM/Carbone)

```python
from src.monitoring import ResourceTracker, CarbonCalculator, MetricsLogger

# Suivi CPU/RAM pendant une operation
with ResourceTracker() as tracker:
    result = model.predict(data)
print(f"CPU: {tracker.usage.cpu_percent_avg:.1f}%")
print(f"RAM peak: {tracker.usage.ram_peak_mb:.0f} MB")

# Estimation empreinte carbone (CodeCarbon)
with CarbonCalculator() as calc:
    result = model.predict(data)
print(f"Emissions: {calc.metrics.emissions_kg:.6f} kg CO2")
print(f"Energie: {calc.metrics.energy_kwh:.6f} kWh")

# Logging des metriques de requetes (JSONL)
logger = MetricsLogger()
with logger.start_request() as timer:
    result = model.predict(data)
# Metriques sauvegardees dans reports/metrics/
```

#### Stack Docker (Prometheus + Grafana + Gatus)

```bash
# Lancer la stack monitoring
cd docker
docker compose --profile monitoring up -d

# Arreter la stack
docker compose --profile monitoring down
```

> **Troubleshooting :** L'erreur NVIDIA (`nvidia-container-cli: initialization error`) concerne
> uniquement le container `app` (GPU requis) et n'affecte pas le monitoring.
> Si Gatus ou Grafana restent en etat "Created" sans demarrer :
> ```bash
> # Demarrer manuellement un container bloque
> docker start docker-gatus-1
> docker start docker-grafana-1
>
> # En cas de probleme persistant, recreer la stack
> docker compose --profile monitoring down -v
> docker compose --profile monitoring up -d
>
> # Verifier l'etat des containers
> docker compose --profile monitoring ps
> ```

| Service | URL | Description |
|---|---|---|
| Prometheus | http://localhost:49090 | Collecte de metriques (CPU/RAM/containers) |
| Grafana | http://localhost:43000 | Dashboards visuels (`admin`/`admin`) |
| Node Exporter | http://localhost:49100 | Metriques systeme |
| cAdvisor | http://localhost:48080 | Metriques containers Docker |
| Gatus | http://localhost:48081 | Health checks & uptime |

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
