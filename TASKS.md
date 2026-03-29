# TASKS - Travel Order Resolver

> Liste exhaustive des taches pour un projet d'excellence.
> Objectif: Depasser les attentes en qualite, gestion de projet et livrables, effectuer tous les bonus.

---

## LEGENDE

- `[P0]` Critique - Requis pour validation
- `[P1]` Important - Attendu
- `[P2]` Bonus explicite du cahier des charges
- `[P3]` Excellence

---

## 1. INFRASTRUCTURE & SETUP PROJET

### 1.1 Initialisation Repository [P0]

- [x] Creer structure de dossiers complete
- [x] Initialiser pyproject.toml avec Poetry
- [x] Configurer .gitignore complet (Python, ML, IDE)
- [x] Creer requirements.txt pour compatibilite pip
- [x] Ecrire README.md professionnel avec badges
- [x] Ajouter LICENSE (MIT)
- [x] Creer CONTRIBUTING.md
- [x] Creer CHANGELOG.md

### 1.2 Configuration Developpement [P1]

- [x] Configurer pre-commit hooks (black, isort, flake8, mypy)
- [ ] Setup .editorconfig
- [x] Creer Makefile avec commandes utiles
- [x] Configurer pytest avec coverage
- [ ] Setup logging structure (loguru ou logging)
- [x] Creer fichier .env.example

### 1.3 CI/CD [P1]

- [x] GitHub Actions: tests automatiques sur PR
- [x] GitHub Actions: linting automatique
- [x] GitHub Actions: build et validation
- [x] GitHub Actions: coverage report
- [x] Configurer branch protection rules
- [x] Template issues (bug, feature, task)
- [x] Template pull request

### 1.4 Containerisation [P2]

- [x] Dockerfile pour l'application
- [x] Dockerfile.training pour entrainement GPU
- [x] docker-compose.yml pour stack complete
- [ ] Documentation deployment Docker

### 1.5 Cloud Deployment [P2]

- [ ] Configuration Azure/GCP/AWS
- [ ] Scripts de deploiement
- [ ] Documentation deployment cloud

---

## 2. GESTION DES DONNEES

### 2.1 Import Donnees SNCF [P0]

- [x] Telecharger liste des gares SNCF (open data)
- [x] Telecharger horaires/lignes SNCF
- [x] Parser JSON gares -> base de donnees interne
- [x] Parser JSON lignes/connexions (structure graphe) — `TrainGraph` construit le graphe depuis les lignes
- [x] Creer mapping ville <-> gare(s)
- [x] Gerer les alias de gares (Paris-Lyon, Paris Gare de Lyon)
- [x] Normaliser les noms (accents, tirets, majuscules)
- [x] Tests unitaires pour le parsing

### 2.2 Creation Dataset NLP [P0]

- [x] Definir schema du dataset (sentence, intent, departure, destination, intermediate)
- [x] Creer ~100 templates de phrases variees (60+ templates implementes)
- [x] Generer phrases avec combinaisons de gares (~10000 phrases)
- [x] Inclure phrases invalides (NOT_TRIP, NOT_FRENCH, UNKNOWN)
- [x] Inclure fautes d'orthographe courantes
- [x] Inclure variations sans majuscules
- [x] Inclure variations sans accents
- [x] Inclure variations avec prenoms-villes (Albert, Paris comme prenom)
- [x] Split train/val/test (70/15/15)
- [ ] Creer dataset de cas limites (edge_cases.csv)
- [x] Documenter le processus de creation (DATASET.md)
- [x] Script de validation du dataset
- [ ] Export formats compatibles HuggingFace

### 2.3 Augmentation de Donnees [P1]

- [ ] Augmentation par synonymes
- [x] Augmentation par fautes de frappe aleatoires
- [x] Augmentation par variation de casse
- [ ] Back-translation (FR->EN->FR)
- [x] Scripts d'augmentation reproductibles

### 2.4 Dataset Audio (Bonus) [P2]

- [ ] Collecter/generer echantillons audio
- [ ] Creer dataset de test speech-to-text
- [ ] Documenter format et sources

### 2.5 Collaboration Inter-groupes [P1]

- [x] Definir format d'echange de datasets
- [x] Contribuer au dataset collectif
- [x] Integrer dataset des autres groupes
- [x] Documenter les contributions

---

## 3. MODULE NLP (COEUR DU PROJET)

### 3.1 Architecture NLP [P0]

- [x] Definir interface abstraite BaseModel (IntentClassifier, EntityExtractor, PostProcessor ABCs)
- [x] Implementer pipeline NLP modulaire (src/nlp/intent/, entity/, post/)
- [ ] Systeme de configuration par YAML/JSON
- [ ] Logging des predictions et metriques

### 3.2 Preprocesseur [P0]

- [x] Normalisation unicode
- [x] Tokenization
- [x] Gestion des majuscules/minuscules
- [x] Gestion des accents
- [x] Gestion des tirets et apostrophes
- [x] Nettoyage caracteres speciaux
- [x] Tests unitaires preprocesseur

### 3.3 Classification d'Intention [P0]

- [x] Classifier binaire: voyage vs non-voyage
- [x] Detection de langue (francais requis)
- [x] Codes d'erreur: NOT_TRIP, NOT_FRENCH, UNKNOWN
- [x] Tests unitaires classification

### 3.4 Extraction d'Entites (NER) [P0]

- [x] Extraction DEPARTURE
- [x] Extraction DESTINATION
- [x] Distinction correcte depart/destination
- [x] Gestion ordre variable dans la phrase
- [x] Tests unitaires NER

### 3.5 Arrêts Intermediaires [P2]

- [x] Extraction INTERMEDIATE (via, en passant par)
- [x] Gestion de plusieurs intermediaires
- [x] Tests unitaires intermediaires

### 3.6 Matching de Gares [P0]

- [x] Matching exact
- [x] Matching fuzzy (Levenshtein, phonetique)
- [ ] Gestion homonymes (Paris ville vs Paris prenom)
- [ ] Desambiguation contextuelle
- [ ] Score de confiance pour chaque match
- [ ] Evaluer tradeoffs seuil fuzzy (80/85/90%) - typo detection vs faux positifs
- [ ] Evaluer impact des samples dest-only sur metriques departure (filtrer ou reporter separement)
- [x] Tests unitaires matching

### 3.7 Modele Baseline (Regex) [P0]

- [x] Implementation rules-based
- [x] Regex pour patterns courants
- [x] Dictionnaire de gares
- [ ] Documenter les limites
- [x] Tests et metriques baseline

### 3.8 Modele SpaCy [P1]

- [x] Integration fr_dep_news_trf
- [x] Extraction NER avec SpaCy
- [x] Post-processing pour depart/destination
- [ ] Fine-tuning SpaCy NER (optionnel)
- [x] Tests et metriques SpaCy

### 3.9 Modele CamemBERT [P1]

- [x] Chargement CamemBERT depuis HuggingFace
- [x] Fine-tuning pour NER custom (B-DEP, I-DEP, B-DEST, I-DEST, B-STEP, I-STEP)
- [x] Intent classification derivee du NER (entities found → TRIP)
- [x] Pipeline d'inference (`CamembertNERModel` + `CamembertEntityExtractor`)
- [x] Partage du modele 420MB entre entity + intent via `create_camembert_components()`
- [x] Tests et metriques CamemBERT (fine-tuned NER)

### 3.10 Modele Flan-T5 / Seq2Seq [P1]

- [x] Chargement Flan-T5 (base ou small)
- [x] Prompt engineering pour extraction
- [x] Fine-tuning seq2seq (Google Colab)
- [x] Pipeline d'inference
- [x] Tests et metriques Flan-T5

### 3.11 Fine-tuning LoRA/QLoRA [P2]

- [ ] Setup PEFT pour fine-tuning efficient
- [ ] Fine-tuning avec LoRA
- [ ] Quantization pour inference rapide
- [ ] Comparaison avec fine-tuning classique

### 3.12 Modele Ensemble [P3]

- [x] Combiner predictions de plusieurs modeles
- [ ] Voting ou stacking
- [ ] Selection dynamique du meilleur modele
- [x] Tests et metriques ensemble

### 3.13 Support Multilingue [P2]

- [x] Detection de langue automatique (RegexLanguageDetector + LangdetectLanguageDetector)
- [x] Support anglais (optionnel) - English patterns in RegexEntityExtractor
- [ ] Support allemand (optionnel)
- [ ] Support espagnol (optionnel)

---

## 4. MODULE SPEECH-TO-TEXT (BONUS)

### 4.1 Integration Whisper [P2]

- [x] Chargement modele Whisper (base) — lazy loading au premier appel
- [x] Transcription audio -> texte — endpoint `POST /api/speech/transcribe`
- [x] Mode offline obligatoire — Whisper tourne localement
- [x] Gestion formats audio (wav, mp3, webm, etc.)
- [x] Tests unitaires transcription (~50 tests)

### 4.2 Pipeline Audio Complet [P2]

- [x] Audio -> Texte -> NLP -> Route — integre dans le frontend (bouton micro)
- [x] Gestion erreurs transcription
- [x] Feedback de confiance (score Whisper retourne)

### 4.3 Ameliorations Audio [P3]

- [ ] Reduction de bruit
- [ ] Detection de silence
- [ ] Streaming audio temps reel

---

## 5. MODULE PATHFINDING

### 5.1 Structure de Graphe [P0]

- [x] Classe Graph avec noeuds (gares) et aretes (connexions) (`TrainGraph`)
- [x] Chargement depuis donnees SNCF (JSON gares + lignes) — 2778 noeuds, 2701 aretes
- [x] Poids: distance (km) avec optimisation LGV (x3 faster)
- [x] Tests unitaires structure graphe (94% coverage)

### 5.2 Algorithme Dijkstra [P0]

- [x] Implementation via NetworkX (production-ready)
- [x] Implementation from scratch (heapq) — `algorithms/dijkstra.py`
- [x] Comprendre et documenter la complexite
- [x] Retourner chemin + distance totale
- [x] Tests unitaires Dijkstra (14 tests pathfinding)

### 5.3 Algorithme A\* [P1]

- [x] Implementation from scratch — `algorithms/astar.py` (heuristique geodesique / 3)
- [x] Heuristique basee sur distance geographique (geodesic)
- [x] Comparaison avec Dijkstra (les deux disponibles, meme resultat optimal)
- [x] Tests unitaires A\*

### 5.4 Gestion Intermediaires [P2]

- [x] Route avec contrainte de passage (chainage A* entre waypoints)
- [x] Optimisation multi-etapes (segments A→B→C)
- [x] MOA* multi-objectif (temps, distance, correspondances) — `algorithms/moa_star.py`

### 5.5 Temps d'Attente [P2]

- [ ] Integration temps d'attente en correspondance
- [ ] Optimisation incluant les attentes
- [ ] Tests unitaires temps d'attente

### 5.6 Multi-Transports [P2]

- [ ] Differencier TGV, TER, Intercites
- [ ] Preferences utilisateur (rapidite vs cout)
- [ ] Filtrage par type de train

### 5.7 Visualisation Graphe [P3]

- [ ] Export graphe pour visualisation
- [ ] Integration avec neo4j (optionnel)
- [x] Carte interactive des routes (`MapVisualizer` avec Folium)

---

## 6. INTERFACE & API

### 6.1 CLI Principal [P0]

- [x] Lecture depuis stdin (`TravelOrderResolver`)
- [ ] Lecture depuis fichier
- [ ] Lecture depuis URL
- [x] Sortie format specifie (sentenceID,Departure,Destination)
- [x] Mode interactif (`python -m src.main`)
- [x] Arguments et options (--help, --extractor)
- [ ] Tests e2e CLI

### 6.2 API REST [P2]

- [x] FastAPI application — `src/api/main.py` avec 7 routers
- [x] Endpoints NLP (`/api/nlp/language`, `/api/nlp/intent`, `/api/nlp/entities`)
- [x] Endpoint pathfinding (`/api/pathfinding/route`, `/api/pathfinding/stations`)
- [x] Endpoint pipeline complet (`/api/resolve`)
- [x] Endpoint speech-to-text (`/api/speech/transcribe`)
- [x] Endpoints evaluation et monitoring
- [x] Schemas Pydantic
- [x] Documentation OpenAPI auto-generee (`/docs`)

### 6.3 Interface Web Demo [P3]

- [x] Frontend Vue 3 + TypeScript + Tailwind CSS
- [x] Input texte et audio (speech-to-text via micro navigateur)
- [x] Visualisation des resultats NLP (entites, intent, langue)
- [x] Affichage de la route sur carte Leaflet interactive
- [x] Page evaluation avec progress bar et matrices de confusion
- [x] Page monitoring (CPU, RAM, latence, carbone)
- [x] Selection du modele NLP dans le chat

---

## 7. MONITORING & OBSERVABILITE

### 7.1 Metriques de Performance [P0]

- [x] Precision, Recall, F1-Score
- [x] Accuracy globale
- [x] Metriques par categorie (intent, departure, destination)
- [x] Matrice de confusion — `src/evaluation/confusion.py`
- [x] Export metriques JSON/CSV

### 7.2 Monitoring Ressources [P2]

- [x] Tracking CPU par requete — `src/monitoring/resource_tracker.py` + psutil
- [x] Tracking RAM par requete
- [x] Tracking GPU utilization par requete (si GPU dispo)
- [x] Tracking temps d'execution — `MetricsLogger`
- [x] Tracking empreinte carbone par requete — `src/monitoring/carbon_calculator.py`

### 7.3 Empreinte Carbone [P2]

- [x] Estimation CO2 par requete — CodeCarbon integre
- [ ] Estimation CO2 pour l'entrainement
- [x] Integration codecarbon
- [x] Endpoint `/api/monitoring/carbon`

### 7.4 Experiment Tracking [P1]

- [x] Integration MLflow — service Docker configure
- [ ] Logging des hyperparametres
- [ ] Logging des metriques d'entrainement
- [ ] Versioning des modeles
- [ ] Comparaison des runs

---

## 8. EVALUATION & BENCHMARKING

### 8.1 Benchmark Multi-Modeles [P1]

- [x] Script de benchmark automatise (`evaluation/evaluate_all.py`)
- [x] Comparaison tous les modeles sur meme dataset
- [x] Tableau comparatif (accuracy, latence, taille) - 5 tables dans README
- [x] Export JSON des metriques (`--output-json`)
- [ ] Graphiques de comparaison
- [x] Selection du meilleur modele (Regex + CamemBERT + Fuzzy)

### 8.2 Analyse d'Erreurs [P1]

- [ ] Categorisation des erreurs
- [ ] Exemples d'erreurs typiques
- [ ] Suggestions d'amelioration
- [ ] Notebook d'analyse detaillee

### 8.3 Robustesse [P1]

- [x] Tests avec fautes d'orthographe (3219 samples evalues)
- [x] Tests sans majuscules (7904 samples evalues)
- [ ] Tests sans accents
- [x] Tests avec bruit (mots parasites)
- [ ] Tests adversariaux

### 8.4 Etudes d'Ablation [P3]

- [ ] Impact de chaque composant
- [ ] Impact de la taille du dataset
- [ ] Impact des hyperparametres
- [ ] Documentation des resultats

---

## 9. DOCUMENTATION

### 9.1 Documentation Technique [P0]

- [x] Architecture complete du systeme
- [x] Description du pipeline NLP
- [x] Description des algorithmes de pathfinding
- [x] Diagrammes (Mermaid/PlantUML)

### 9.2 Documentation Entrainement [P0]

- [ ] Processus de creation du dataset
- [ ] Procedure d'entrainement
- [ ] Hyperparametres finaux
- [ ] Resultats des experimentations

### 9.3 Exemple Detaille [P0]

- [ ] Trace complete d'une phrase a travers le systeme
- [ ] Visualisation de chaque etape
- [ ] Explication des decisions du modele

### 9.4 Rapport PDF Final [P0]

- [ ] Minimum 5 pages utiles
- [ ] Figures et graphiques (pas de code)
- [ ] Comparaison des modeles
- [ ] Conclusions et perspectives
- [ ] Mise en page professionnelle

### 9.5 Documentation API [P2]

- [x] OpenAPI/Swagger auto-genere — FastAPI `/docs`
- [x] Contrat API documente — `docs/API_CONTRACT.md`
- [ ] Postman collection

### 9.6 Notebooks Jupyter [P1]

- [ ] Exploration des donnees
- [ ] Prototypage des modeles
- [ ] Visualisation de l'attention
- [ ] Demo interactive

---

## 10. TESTS & QUALITE

### 10.1 Tests Unitaires [P0]

- [x] Tests preprocesseur — `test_nlp.py`
- [x] Tests NER — `test_entity_extractor.py`, `test_camembert_ner.py`, `test_flant5.py`
- [x] Tests classification — `test_intent_classifier.py`
- [x] Tests matching gares — `test_fuzzy_matcher.py`, `test_data_parsing.py`
- [x] Tests pathfinding — `test_pathfinding.py` (14 tests)
- [ ] Coverage > 80% (actuellement 39% global, >90% sur composants critiques)

### 10.2 Tests Integration [P1]

- [ ] Tests pipeline NLP complet
- [ ] Tests NLP -> Pathfinding
- [ ] Tests API endpoints

### 10.3 Tests E2E [P1]

- [ ] Tests CLI complet
- [ ] Tests avec fichiers reels
- [ ] Tests de performance

### 10.4 Qualite Code [P1]

- [ ] Type hints partout
- [ ] Docstrings Google style
- [ ] Pas de code duplique
- [ ] Respect PEP8
- [x] Black sans erreurs
- [x] iSort sans erreurs
- [x] Flake8 sans erreurs
- [x] Mypy sans erreurs

---

## 11. GESTION DE PROJET

### 11.1 Organisation Equipe [P1]

- [x] Definir roles (NLP Lead, Data, Pathfinding, Infra)
- [x] Definir canaux de communication
- [x] Planning des reunions
- [x] Definition of Done pour les taches

### 11.2 Gestion Git [P1]

- [x] Branching strategy (feature branches)
- [ ] Commits reguliers de tous les membres
- [x] Messages de commit conventionnels
- [x] Code review obligatoire
- [x] Pas de force push sur main

### 11.3 Suivi Avancement [P1]

- [ ] Board Kanban (GitHub Projects)
- [ ] Milestones pour les deadlines
- [x] Daily/Weekly sync
- [ ] Retrospectives

### 11.4 Gestion des Risques [P3]

- [ ] Identification des risques techniques
- [ ] Plans de mitigation
- [ ] Backup strategies

---

## 12. PRESENTATION ORALE

### 12.1 Preparation Slides [P0]

- [ ] Slides sobres et professionnelles
- [ ] Architecture claire
- [ ] Demo live
- [ ] Resultats et metriques
- [ ] Questions anticipees

### 12.2 Preparation Equipe [P0]

- [ ] Tous les membres peuvent expliquer le code
- [ ] Tous comprennent les algorithmes
- [ ] Tous connaissent les metriques
- [ ] Repetition generale

### 12.3 Demo [P1]

- [ ] Demo CLI fonctionnelle
- [ ] Demo avec cas difficiles
- [ ] Demo speech-to-text (si bonus)
- [ ] Gestion des erreurs en live

---

## 13. BONUS SUPPLEMENTAIRES

### 13.1 Interpretabilite [P3]

- [ ] Visualisation de l'attention
- [ ] Explication des predictions (LIME/SHAP)
- [ ] Heatmaps token-attention

### 13.2 Optimisation Inference [P3]

- [ ] Quantization des modeles
- [ ] ONNX export
- [x] Batch processing (CamemBERT classify_batch, SpaCy extract_batch)
- [x] Caching intelligent (FuzzyMatcher \_match_cache, pre-built indices)

### 13.3 Tests de Charge [P3]

- [ ] Benchmark avec 1000+ requetes
- [ ] Identification des goulots d'etranglement
- [ ] Optimisation performance

### 13.4 Monitoring Production [P3]

- [x] Prometheus metrics — docker-compose avec scraping
- [x] Grafana dashboards — provisionnes dans `src/monitoring/grafana/`
- [x] Health checks — Gatus configure

### 13.5 Documentation Video [P3]

- [ ] Video de demonstration
- [ ] Tutorial d'installation
- [ ] Walkthrough du code

---

## CHECKLIST FINALE PRE-RENDU

### Livrables Obligatoires

- [x] Code source complet et fonctionnel
- [x] Module NLP isole et testable (4 backends)
- [x] Module Pathfinding fonctionnel (3 algorithmes)
- [x] Dataset avec train/val/test splits
- [ ] Documentation technique PDF
- [x] README complet

### Qualite

- [x] Tous les tests passent (234/234)
- [ ] Coverage > 80% (39% global)
- [x] Pas d'erreurs mypy
- [x] Code formate (black)
- [x] Git historique propre

### Bonus Implementes

- [x] Speech-to-text offline (Whisper)
- [x] Arrêts intermediaires
- [x] Benchmark multi-modeles
- [x] Monitoring ressources (CPU, RAM, GPU, carbone)
- [x] API REST (FastAPI, 7 routers)
- [x] Interface demo (Vue 3 + carte Leaflet)

### Presentation

- [ ] Slides pretes
- [ ] Demo fonctionnelle
- [ ] Equipe preparee

---

_Derniere mise a jour: 2026-03-27 (v0.5.0 - Demo cleanup: fix Flan-T5, SpaCy, Docker, API, fuzzy matching)_
