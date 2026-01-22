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
- [ ] Parser JSON lignes/connexions (structure graphe)
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
- [ ] Fine-tuning pour classification d'intention
- [ ] Fine-tuning pour NER custom (B-DEP, I-DEP, B-DEST, I-DEST)
- [x] Pipeline d'inference (`CamembertZeroShotExtractor`)
- [x] Tests et metriques CamemBERT (zero-shot: 6.7% accuracy)

### 3.10 Modele Flan-T5 / Seq2Seq [P1]
- [ ] Chargement Flan-T5 (base ou small)
- [ ] Prompt engineering pour extraction
- [ ] Fine-tuning seq2seq
- [ ] Pipeline d'inference
- [ ] Tests et metriques Flan-T5

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
- [ ] Detection de langue automatique
- [ ] Support anglais (optionnel)
- [ ] Support allemand (optionnel)
- [ ] Support espagnol (optionnel)

---

## 4. MODULE SPEECH-TO-TEXT (BONUS)

### 4.1 Integration Whisper [P2]
- [ ] Chargement modele Whisper (small/base)
- [ ] Transcription audio -> texte
- [ ] Mode offline obligatoire
- [ ] Gestion formats audio (wav, mp3, etc.)
- [ ] Tests unitaires transcription

### 4.2 Pipeline Audio Complet [P2]
- [ ] Audio -> Texte -> NLP -> Route
- [ ] Gestion erreurs transcription
- [ ] Feedback de confiance

### 4.3 Ameliorations Audio [P3]
- [ ] Reduction de bruit
- [ ] Detection de silence
- [ ] Streaming audio temps reel

---

## 5. MODULE PATHFINDING

### 5.1 Structure de Graphe [P0]
- [x] Classe Graph avec noeuds (gares) et aretes (connexions) (`TrainGraph`)
- [x] Chargement depuis donnees SNCF (CSV gares + lignes)
- [x] Poids: distance (km) avec optimisation LGV (x3 faster)
- [ ] Tests unitaires structure graphe

### 5.2 Algorithme Dijkstra [P0]
- [x] Implementation via NetworkX (production-ready)
- [ ] Implementation from scratch (pas de librairie) - pour comprendre
- [x] Comprendre et documenter la complexite
- [x] Retourner chemin + distance totale
- [ ] Tests unitaires Dijkstra

### 5.3 Algorithme A* [P1]
- [ ] Implementation from scratch
- [ ] Heuristique basee sur distance geographique
- [ ] Comparaison avec Dijkstra
- [ ] Tests unitaires A*

### 5.4 Gestion Intermediaires [P2]
- [ ] Route avec contrainte de passage
- [ ] Optimisation multi-etapes
- [ ] Tests unitaires intermediaires

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
- [ ] FastAPI application
- [ ] Endpoint /parse pour NLP
- [ ] Endpoint /route pour pathfinding
- [ ] Endpoint /full pour pipeline complet
- [ ] Schemas Pydantic
- [ ] Documentation OpenAPI auto-generee
- [ ] Tests API

### 6.3 Interface Web Demo [P3]
- [ ] Interface Gradio ou Streamlit
- [ ] Input texte et audio
- [ ] Visualisation des resultats
- [ ] Affichage de la route sur carte

---

## 7. MONITORING & OBSERVABILITE

### 7.1 Metriques de Performance [P0]
- [x] Precision, Recall, F1-Score
- [x] Accuracy globale
- [x] Metriques par categorie (intent, departure, destination)
- [ ] Matrice de confusion
- [x] Export metriques JSON/CSV

### 7.2 Monitoring Ressources [P2]
- [ ] Tracking CPU par requete
- [ ] Tracking RAM par requete
- [x] Tracking temps d'execution
- [ ] Tracking GPU utilization (si applicable)

### 7.3 Empreinte Carbone [P2]
- [ ] Estimation CO2 par requete
- [ ] Estimation CO2 pour l'entrainement
- [ ] Integration codecarbon ou equivalent
- [ ] Rapport d'impact environnemental

### 7.4 Experiment Tracking [P1]
- [ ] Integration MLflow ou Weights & Biases
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
- [ ] Description des algorithmes de pathfinding
- [ ] Diagrammes (Mermaid/PlantUML)

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
- [ ] OpenAPI/Swagger auto-genere
- [ ] Exemples d'utilisation
- [ ] Postman collection

### 9.6 Notebooks Jupyter [P1]
- [ ] Exploration des donnees
- [ ] Prototypage des modeles
- [ ] Visualisation de l'attention
- [ ] Demo interactive

---

## 10. TESTS & QUALITE

### 10.1 Tests Unitaires [P0]
- [ ] Tests preprocesseur
- [ ] Tests NER
- [ ] Tests classification
- [ ] Tests matching gares
- [ ] Tests pathfinding
- [ ] Coverage > 80%

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
- [x] Caching intelligent (FuzzyMatcher _match_cache, pre-built indices)

### 13.3 Tests de Charge [P3]
- [ ] Benchmark avec 1000+ requetes
- [ ] Identification des goulots d'etranglement
- [ ] Optimisation performance

### 13.4 Monitoring Production [P3]
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Alerting

### 13.5 Documentation Video [P3]
- [ ] Video de demonstration
- [ ] Tutorial d'installation
- [ ] Walkthrough du code

---

## CHECKLIST FINALE PRE-RENDU

### Livrables Obligatoires
- [ ] Code source complet et fonctionnel
- [ ] Module NLP isole et testable
- [ ] Module Pathfinding fonctionnel
- [ ] Dataset avec train/val/test splits
- [ ] Documentation technique PDF
- [ ] README complet

### Qualite
- [ ] Tous les tests passent
- [ ] Coverage > 80%
- [ ] Pas d'erreurs mypy
- [ ] Code formate (black)
- [ ] Git historique propre

### Bonus Implementes
- [ ] Speech-to-text offline
- [x] Arrêts intermediaires
- [x] Benchmark multi-modeles
- [ ] Monitoring ressources
- [ ] API REST
- [ ] Interface demo

### Presentation
- [ ] Slides pretes
- [ ] Demo fonctionnelle
- [ ] Equipe preparee

---

*Derniere mise a jour: 2025-01-22 (v0.3.0 - 100k STT Dataset with Intermediate Stops)*
