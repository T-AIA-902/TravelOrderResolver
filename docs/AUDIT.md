# Audit technique — Travel Order Resolver

> **Date** : 27 mars 2026
> **Branche** : `feature/demo-cleanup` (dev + lastBranch)
> **Contexte** : Reprise du projet pour préparation démo finale

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture](#2-architecture)
3. [Stack technique](#3-stack-technique)
4. [Composants — État détaillé](#4-composants--état-détaillé)
   - [Pipeline NLP](#41-pipeline-nlp)
   - [Pathfinding](#42-pathfinding)
   - [API REST](#43-api-rest)
   - [Frontend](#44-frontend)
   - [Speech-to-text](#45-speech-to-text)
   - [Évaluation](#46-évaluation)
   - [Monitoring](#47-monitoring)
   - [Docker](#48-docker)
5. [Tests](#5-tests)
6. [Données SNCF](#6-données-sncf)
7. [Modèles ML](#7-modèles-ml)
8. [Problèmes identifiés](#8-problèmes-identifiés)
9. [Reste à faire pour la démo](#9-reste-à-faire-pour-la-démo)
10. [Commandes utiles](#10-commandes-utiles)

---

## 1. Vue d'ensemble

Travel Order Resolver est un système NLP qui transforme des demandes de voyage en
langage naturel français (texte ou voix) en itinéraires ferroviaires optimaux sur le
réseau SNCF.

**Fonctionnalités principales :**

- Extraction d'entités (villes de départ, arrivée, étapes) depuis du texte libre
- Détection de langue (français / anglais)
- Classification d'intention (voyage / pas un voyage)
- Calcul d'itinéraires sur le réseau ferré français (3 000+ gares)
- Interface web avec chat, carte interactive, évaluation et monitoring
- Reconnaissance vocale (speech-to-text via Whisper)

---

## 2. Architecture

```
┌─────────────────────────┐     HTTP      ┌──────────────────────────────┐
│   Frontend Vue 3        │ ◄──────────►  │   API FastAPI                │
│   (port 5173 / 3000)    │    /api/*     │   (port 8000)               │
├─────────────────────────┤               ├──────────────────────────────┤
│ - Chat + carte Leaflet  │               │ /api/health                  │
│ - Dashboard             │               │ /api/resolve    ← principal  │
│ - Évaluation            │               │ /api/nlp/*      ← debug     │
│ - Monitoring            │               │ /api/speech/transcribe       │
│ - Rapports              │               │ /api/pathfinding/*           │
│ - Dataset (placeholder) │               │ /api/evaluation/*            │
└─────────────────────────┘               │ /api/monitoring/*            │
                                          └──────┬───────────────────────┘
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          ▼                      ▼                      ▼
                  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
                  │ Pipeline NLP │     │ Pathfinding   │     │ Whisper STT  │
                  ├──────────────┤     ├──────────────┤     └──────────────┘
                  │ Prétraitement│     │ Dijkstra      │
                  │ Langue       │     │ A*            │
                  │ Intention    │     │ MOA*          │
                  │ Entités      │     │ (3000+ gares) │
                  │ Fuzzy match  │     └──────────────┘
                  └──────────────┘

                  4 backends :
                  Regex | SpaCy | CamemBERT | Flan-T5
```

**Flux principal :**

```
Texte utilisateur
  → Prétraitement (normalisation Unicode, accents, espaces)
  → Détection de langue (FRENCH / ENGLISH / UNKNOWN)
  → Classification d'intention (TRIP / NOT_TRIP / UNKNOWN)
  → Extraction d'entités (départ, destination, étapes)
  → Fuzzy matching vers la base SNCF
  → Pathfinding (calcul d'itinéraire sur le graphe)
  → Réponse JSON (itinéraire + métadonnées + carte)
```

---

## 3. Stack technique

### Backend (Python 3.10+)

| Catégorie       | Technologies                                          |
|-----------------|-------------------------------------------------------|
| API             | FastAPI 0.108, Uvicorn 0.25, Pydantic 2.5             |
| NLP / ML        | transformers 4.36, torch 2.1, spaCy 3.7, peft 0.7     |
| Speech          | openai-whisper, sounddevice, soundfile                 |
| Pathfinding     | networkx 3.0, geopy 2.4, shapely 2.0                  |
| Matching        | rapidfuzz 3.5                                          |
| Visualisation   | folium 0.15                                            |
| Monitoring      | mlflow 2.9, codecarbon 2.3, psutil 5.9                 |
| CLI             | typer 0.9, rich 13.7                                   |
| Qualité         | pytest 7.4, black, isort, mypy, flake8                 |

### Frontend (TypeScript)

| Catégorie       | Technologies                                          |
|-----------------|-------------------------------------------------------|
| Framework       | Vue 3.5, Vue Router 4.6, TypeScript 5.9               |
| Build           | Vite 7.3                                               |
| Style           | Tailwind CSS 4.1                                       |
| Graphiques      | ECharts 6.0                                            |
| Cartes          | Leaflet 1.9, vue-leaflet                               |
| Icônes          | Lucide Vue Next                                        |

### Infrastructure

| Catégorie       | Technologies                                          |
|-----------------|-------------------------------------------------------|
| Conteneurs      | Docker, Docker Compose 3.8                             |
| Monitoring      | Prometheus, Grafana, cAdvisor, Gatus                   |
| Reverse proxy   | Nginx Alpine                                           |
| GPU             | CUDA 12.1 (optionnel)                                  |

---

## 4. Composants — État détaillé

### 4.1 Pipeline NLP

**Fichiers principaux :**

| Fichier | Rôle |
|---------|------|
| `src/nlp/pipeline.py` | Orchestrateur principal |
| `src/nlp/preprocessor.py` | Normalisation texte (Unicode, accents, espaces) |
| `src/nlp/interfaces.py` | Classes abstraites (LanguageDetector, IntentClassifier, EntityExtractor) |
| `src/nlp/types.py` | Types : Intent, Language, PredictionResult |

**4 backends d'extraction :**

| Backend | Langue | Intention | Entités | Poids | Notes |
|---------|--------|-----------|---------|-------|-------|
| **Regex** | ✅ Patterns FR/EN | ✅ Mots-clés voyage | ✅ 10+ patterns | 0 | Aucun modèle requis, rapide |
| **SpaCy** | — | — | ✅ NER (LOC, GPE) | 40 MB | Nécessite `fr_core_news_lg` |
| **CamemBERT** | — | ✅ Dérivé du NER | ✅ Token classification (BIO) | 420 MB | Fine-tuné, labels B-DEP/I-DEP/B-DEST/I-DEST/B-STEP/I-STEP |
| **Flan-T5** | — | ✅ Prompt seq2seq | ✅ Prompt seq2seq | ~500 MB | Fine-tuné, meilleur F1 (87%) |

**Post-traitement :**

- `src/nlp/post/fuzzy_post_processor.py` : Matching flou (RapidFuzz, seuil 80%) vers les noms officiels SNCF
- Cache LRU pour les requêtes répétées

**Status : ✅ COMPLET ET FONCTIONNEL**

---

### 4.2 Pathfinding

**Fichiers :**

| Fichier | Rôle |
|---------|------|
| `src/pathfinding/graph.py` | Construction du graphe ferroviaire (NetworkX MultiGraph) |
| `src/pathfinding/route_optimizer.py` | Façade unifiée pour les 3 algorithmes |
| `src/pathfinding/algorithms/dijkstra.py` | Dijkstra from-scratch (heapq) |
| `src/pathfinding/algorithms/astar.py` | A* from-scratch (heuristique géodésique) |
| `src/pathfinding/algorithms/moa_star.py` | MOA* multi-objectif (temps, distance, correspondances) |

**Caractéristiques du graphe :**

- 3 000+ gares voyageurs
- 6 000+ connexions ferroviaires
- Optimisation LGV : poids ÷ 3 pour les lignes à grande vitesse
- Transferts intra-ville : connexions automatiques entre gares d'une même ville (< 8 km)
- Hubs : Paris (15 transferts), Lyon, Marseille, Lille, etc.

**Algorithmes :**

| Algorithme | Complexité | Optimal | Multi-objectif | Étapes intermédiaires |
|------------|-----------|---------|----------------|----------------------|
| Dijkstra | O((V+E)log V) | ✅ | Non | Via chaînage |
| A* | O((V+E)log V) | ✅ | Non | Via chaînage |
| MOA* | Variable | ✅ Pareto | ✅ (temps, dist, corresp.) | Via produit cartésien |

**Status : ✅ COMPLET ET FONCTIONNEL**

---

### 4.3 API REST

**Point d'entrée :** `src/api/main.py`

| Route | Méthode | Description | Status |
|-------|---------|-------------|--------|
| `/api/health` | GET | État du serveur, modèles chargés, device | ✅ |
| `/api/resolve` | POST | Pipeline complet : NLP + pathfinding | ✅ |
| `/api/nlp/language` | POST | Détection de langue (tous modèles) | ✅ |
| `/api/nlp/intent` | POST | Classification d'intention (tous modèles) | ✅ |
| `/api/nlp/entities` | POST | Extraction d'entités (tous modèles) | ✅ |
| `/api/speech/transcribe` | POST | Audio → texte (Whisper) | ✅ |
| `/api/pathfinding/stations` | GET | Recherche de gares par nom | ✅ |
| `/api/pathfinding/route` | POST | Calcul d'itinéraire avec étapes | ✅ |
| `/api/evaluation/reports` | GET | Liste des rapports d'évaluation | ✅ |
| `/api/evaluation/run` | POST | Lancer une évaluation (async) | ✅ |
| `/api/evaluation/status/{id}` | GET | Progression d'une évaluation | ✅ |
| `/api/monitoring/metrics` | GET | Historique des requêtes + stats | ✅ |
| `/api/monitoring/resources` | GET | CPU / RAM / GPU en temps réel | ✅ |
| `/api/monitoring/carbon` | GET | Empreinte carbone cumulée | ✅ |

**Endpoint principal — `POST /api/resolve` :**

```json
// Requête
{
  "text": "Je veux aller de Paris à Lyon en passant par Dijon",
  "intent_model": "camembert",
  "entity_model": "camembert",
  "use_fuzzy": true
}

// Réponse
{
  "nlp": {
    "language": { "label": "FRENCH", "confidence": 0.95 },
    "intent": { "label": "TRIP", "confidence": 0.9 },
    "entities": {
      "departure": "Paris",
      "destination": "Lyon",
      "intermediate": ["Dijon"]
    }
  },
  "pathfinding": {
    "segments": [...],
    "total_distance_km": 462,
    "num_transfers": 1
  },
  "latency_ms": 342
}
```

**CORS :** `localhost:5173` (dev) et `localhost:3000` (prod) — hardcodé.

**Status : ✅ COMPLET ET FONCTIONNEL**

---

### 4.4 Frontend

**Framework :** Vue 3 + TypeScript + Vite + Tailwind CSS

**Pages :**

| Page | Route | Status | Description |
|------|-------|--------|-------------|
| Dashboard | `/` | ✅ Complet | Accueil avec navigation vers les fonctionnalités |
| Chat | `/chat` | ✅ Complet | Saisie texte/voix, sélection de modèle, résultats NLP, carte |
| Évaluation | `/evaluation` | ✅ Complet | Lancement d'évals, progress bar, matrices de confusion |
| Monitoring | `/monitoring` | ✅ Complet | Jauges CPU/RAM/GPU, latence, carbone |
| Rapports | `/rapports` | ⚠️ Partiel | Tableau de benchmarks **hardcodé**, pas connecté à l'API |
| Dataset | `/dataset` | ❌ Placeholder | "Coming soon" |

**Composables (logique réutilisable) :**

| Composable | Rôle |
|------------|------|
| `useResolve()` | Gestion requête/réponse resolve |
| `useAudioRecorder()` | Enregistrement audio + transcription |
| `useEvaluation()` | Rapports et exécution d'évaluations |
| `useMapRoute()` | Segments de route pour Leaflet |
| `useMonitoring()` | Métriques, ressources, carbone |

**Client API :** `src/frontend/src/api/client.ts` — `apiGet`, `apiPost`, `apiPostForm` avec proxy Vite vers `localhost:8000`.

**Status : ✅ FONCTIONNEL (5/6 pages)**

---

### 4.5 Speech-to-text

| Couche | Implémentation | Status |
|--------|---------------|--------|
| Frontend | Web Audio API + MediaRecorder → WebM | ✅ |
| API | `POST /api/speech/transcribe` (UploadFile) | ✅ |
| Backend | Whisper model `base` (~140 MB), chargement lazy | ✅ |

**Flux :** Clic micro → enregistrement → envoi WebM → Whisper → texte transcrit → auto-envoi au chat → NLP → réponse.

**Status : ✅ COMPLET ET INTÉGRÉ**

---

### 4.6 Évaluation

**Fichiers :** `src/evaluation/`

| Évaluateur | Métriques |
|------------|-----------|
| Entity | Precision, Recall, F1 par champ (départ, destination) |
| Intent | Accuracy, Precision, Recall, F1 |
| Language | Accuracy, matrice de confusion |
| Combined | Pipeline de bout en bout |

**Model factory** (`src/evaluation/model_factory.py`) : instanciation centralisée, partage du modèle CamemBERT (évite double chargement 420 MB).

**Status : ✅ COMPLET**

---

### 4.7 Monitoring

| Métrique | Source |
|----------|--------|
| CPU / RAM / GPU | psutil, temps réel |
| Latence requêtes | MetricsLogger interne |
| Empreinte carbone | CodeCarbon (CO2e par requête) |
| Infrastructure | Prometheus → Grafana (via Docker) |

**Status : ✅ COMPLET**

---

### 4.8 Docker

**Fichiers :** `docker/`

| Service | Port | Image | Notes |
|---------|------|-------|-------|
| Backend (`app`) | 8000 | PyTorch 2.1 + CUDA 12.1 | ⚠️ CMD lance le CLI, pas l'API |
| Frontend | 3000 | Node 22 → Nginx Alpine | Multi-stage build |
| Prometheus | 9090 | prom/prometheus:v2.51.0 | Scrape métriques |
| Grafana | 3001 | grafana/grafana:10.4.0 | Dashboards |
| Node Exporter | 9100 | Métriques système | — |
| cAdvisor | 8080 | Métriques conteneurs | — |
| Gatus | 8081 | Health checks | — |

**Nginx** (`docker/nginx-frontend.conf`) : SPA routing + proxy `/api/*` → `app:8000`.

**Status : ⚠️ FONCTIONNEL MAIS CMD BACKEND À CORRIGER**

---

## 5. Tests

**Framework :** pytest 7.4 | **Résultat : 234/234 passent** (~28s)

| Fichier de test | Nb tests | Couverture cible |
|-----------------|----------|------------------|
| `test_nlp.py` | ~40 | Pipeline NLP complet |
| `test_pathfinding.py` | 14 | Dijkstra, A*, graphe |
| `test_data_parsing.py` | 29 | Station DB, normalisation |
| `test_fuzzy_matcher.py` | ~30 | Matching flou |
| `test_entity_extractor.py` | ~30 | Extraction Regex/SpaCy |
| `test_intent_classifier.py` | ~15 | Classification d'intention |
| `test_camembert_ner.py` | 14 | NER CamemBERT (BIO tags) |
| `test_flant5.py` | ~15 | Flan-T5 |
| `test_speech.py` | ~50 | Speech-to-text |

**Couverture par composant :**

| Composant | Couverture |
|-----------|-----------|
| `pathfinding/graph.py` | 94% |
| `speech/` | 94-97% |
| `data/station_database.py` | 82% |
| `nlp/entity/` | 68-96% |
| **Moyenne globale** | **39%** |

**Non testés :** Évaluation CLI (0%), Visualisation (0%).

---

## 6. Données SNCF

**Emplacement :** `datasets/raw/sncf/`

| Fichier | Taille | Contenu |
|---------|--------|---------|
| `gares-de-voyageurs.json` | 579 KB | Métadonnées des gares (GPS, noms) |
| `liste-des-gares.json` | 4.8 MB | Association gare ↔ ligne (pk) |
| `lignes-par-type.json` | 9.3 MB | Définitions des lignes (dont LGV) |
| `tgvmax.json` | 103 MB | Horaires TGV (non utilisé actuellement) |

**Datasets d'entraînement/test :** `datasets/` contient les splits train/test/validation (~1k exemples de base, 100k+ augmentés avec simulation d'erreurs STT).

---

## 7. Modèles ML

| Modèle | Type | Emplacement | Taille | Téléchargement |
|--------|------|-------------|--------|----------------|
| CamemBERT NER | Token classification (fine-tuné) | `models/camembert-ner-retrain/` | 420 MB | `scripts/download_models.sh` |
| Flan-T5 Travel | Seq2seq (fine-tuné) | `models/flan-t5-travel/` | ~500 MB | `scripts/download_models.sh` |
| spaCy fr_core_news_lg | NER pré-entraîné | Via pip | 40 MB | `python -m spacy download fr_core_news_lg` |
| Whisper base | Speech-to-text | Cache HuggingFace | ~140 MB | Auto-téléchargé au premier appel |

**Script de téléchargement :** `scripts/download_models.sh` — référence le repo HuggingFace `Vatt/travel-order-resolver-models`.

**Fallback :** Sans modèles téléchargés, le backend **Regex** fonctionne sans aucune dépendance ML.

---

## 8. Problèmes identifiés

### Corrigés

| # | Problème | Fix | Status |
|---|----------|-----|--------|
| ~~1~~ | Flan-T5 : singleton empêchait la coexistence entity (fine-tuné) + intent (base) | Remplacé par cache par chemin (`get_flan_t5_loader`) | ✅ Corrigé |
| ~~2~~ | Flan-T5 : crash MPS sur Apple Silicon | Forcé CPU (MPS incompatible avec T5 safetensors) | ✅ Corrigé |
| ~~3~~ | Flan-T5 : intent classification cassée (renvoyait des entités) | Modèle base chargé séparément du fine-tuné | ✅ Corrigé |
| ~~4~~ | SpaCy : heuristiques en dur masquaient les limites du modèle | Nettoyé : assignation positionnelle pure (NER générique) | ✅ Corrigé |
| ~~5~~ | Docker backend lançait le CLI au lieu de l'API | CMD remplacée par uvicorn dans docker-compose.yml | ✅ Corrigé |
| ~~6~~ | API : nom de modèle "flant5" ne matchait pas "Flan-T5" | Normalisation des noms (ignore tirets/underscores/casse) | ✅ Corrigé |

### Critique

Aucun problème critique restant.

### Moyen

| # | Problème | Fichier | Impact |
|---|----------|---------|--------|
| 6 | CORS hardcodé (localhost uniquement) | `src/api/main.py:55-61` | Pas déployable en l'état |
| 7 | Page Rapports hardcodée | `src/frontend/` | Données figées, pas les vrais résultats |
| 8 | Page Dataset placeholder | `src/frontend/` | Page vide |

### Mineur

| # | Problème | Fichier | Impact |
|---|----------|---------|--------|
| 9 | Paramètre `language` speech non utilisé | `src/api/routers/speech.py` | Fonctionnel mais paramètre ignoré |
| 10 | Pas de typage erreur côté frontend | `src/frontend/src/api/client.ts` | Erreurs API non parsées |

---

## 9. Reste à faire pour la démo

**Déjà fait :**

- [x] Fix Flan-T5 : singleton → cache par chemin, MPS → CPU, intent utilise modèle base
- [x] Fix SpaCy : suppression des heuristiques en dur, assignation positionnelle pure
- [x] Tests : 234/234 passent
- [x] Vérification modèles : Regex ✅, CamemBERT ✅, SpaCy ✅, Flan-T5 ✅
- [x] API démarre et `/api/health` répond 200 (4 modèles chargés, 2778 gares)
- [x] `/api/resolve` fonctionne avec les 4 modèles (regex, camembert, spacy, flant5)
- [x] Pathfinding trouve des routes (ex: Paris → Marseille)
- [x] Fix Docker backend CMD → uvicorn
- [x] Fix matching noms de modèles dans l'API (flant5 = Flan-T5)
- [x] Frontend : dépendances installées, proxy Vite configuré vers :8000

**Priorité 2 — À tester manuellement :**

- [ ] Lancer API + frontend ensemble et tester le flow dans le navigateur
- [ ] Tester le speech-to-text dans le navigateur
- [ ] Vérifier que l'évaluation se lance depuis le frontend

**Priorité 3 — Polish :**

- [ ] Connecter la page Rapports aux vrais résultats d'évaluation
- [ ] Rendre le CORS configurable par variable d'environnement
- [ ] Nettoyer les fichiers non committés

---

## 10. Commandes utiles

```bash
# Installation
make install              # Dépendances production
make install-dev          # + outils de dev
make install-all          # Tout (ML inclus)

# Lancer le projet
make serve                # API FastAPI sur :8000
cd src/frontend && npm run dev   # Frontend sur :5173

# Tests
make test                 # 233 tests + couverture
make lint                 # Linters (black, isort, flake8, mypy)

# Docker
make docker-compose-up    # Stack complète
make front-run            # Frontend seul (Docker)

# Évaluation
make evaluate             # Benchmark sur jeu de test
make demo-all             # Démo avec tous les extracteurs

# Données
make download-data        # Données SNCF
make generate-dataset     # Dataset de base
make augment-data         # Augmentation STT

# Modèles
bash scripts/download_models.sh   # Télécharger CamemBERT + Flan-T5
python -m spacy download fr_core_news_lg   # Modèle spaCy
```
