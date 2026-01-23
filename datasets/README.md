# Datasets Documentation

> Datasets synthetiques de demandes de voyage ferroviaire en langage naturel.

**Date de generation:** Janvier 2026
**Nombre total d'entrees:** 100 000 par dataset
**Seed de reproductibilite:** 42
**Distribution:** Realiste (optimisee pour systeme ferroviaire francais)

---

## Table des matieres

1. [Objectif](#objectif)
2. [Architecture des datasets](#architecture-des-datasets)
3. [Dataset Base](#dataset-base)
4. [Dataset Augmented (STT)](#dataset-augmented-stt)
5. [Types d'erreurs STT simulees](#types-derreurs-stt-simulees)
6. [Structure commune](#structure-commune)
7. [Distribution des donnees](#distribution-des-donnees)
8. [Scripts utilises](#scripts-utilises)
9. [Exemples](#exemples)
10. [Limites et considerations](#limites-et-considerations)
11. [Regeneration des datasets](#regeneration-des-datasets)

---

## Objectif

Ces datasets ont ete crees pour entrainer et evaluer des modeles NLP capables de :

1. **Classifier l'intention** (TRIP, NOT_TRIP, UNKNOWN)
2. **Detecter la langue** (FRENCH, ENGLISH, SPANISH, GERMAN, ITALIAN, UNKNOWN)
3. **Extraire les entites** (gare de depart, gare d'arrivee, gare intermediaire)

---

## Architecture des datasets

Le projet utilise une architecture a deux niveaux pour permettre des etudes d'ablation:

```
datasets/
├── base/              # Phrases propres (sans erreurs STT)
│   ├── base_dataset_100k.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── augmented/         # Avec erreurs STT simulees
│   ├── stt_dataset_100k.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── scripts/
    ├── generate_base.py   # Generation du dataset base
    └── augment_stt.py     # Augmentation STT
```

### Workflow

1. **Generation** : `generate_base.py` cree le dataset base propre
2. **Augmentation** : `augment_stt.py` lit le dataset base et applique les erreurs STT

### Interet academique

Cette separation permet de :
- Comparer les performances sur donnees propres vs bruitees
- Mesurer l'impact des erreurs STT sur chaque composant
- Realiser des etudes d'ablation rigoureuses

---

## Dataset Base

**Repertoire:** `datasets/base/`

Dataset contenant des phrases **propres**, sans erreurs de transcription. Les phrases sont grammaticalement correctes et bien formatees.

| Fichier | Lignes | Description |
|---------|--------|-------------|
| `base_dataset_100k.csv` | 100 000 | Dataset complet |
| `train.csv` | 69 998 | Entrainement (70%) |
| `val.csv` | 14 997 | Validation (15%) |
| `test.csv` | 15 005 | Test (15%) |

**Format ID:** `BASE000001`

**Utilisation:** Entrainement de modeles sur donnees propres, baseline pour comparaison.

---

## Dataset Augmented (STT)

**Repertoire:** `datasets/augmented/`

Dataset avec des **erreurs STT simulees** appliquees aux phrases du dataset base. Simule des transcriptions realistes de Whisper.

| Fichier | Lignes | Description |
|---------|--------|-------------|
| `stt_dataset_100k.csv` | 100 000 | Dataset complet |
| `train.csv` | 69 998 | Entrainement (70%) |
| `val.csv` | 14 997 | Validation (15%) |
| `test.csv` | 15 005 | Test (15%) |

**Format ID:** `STT000001`

**Utilisation:** Entrainement de modeles robustes aux erreurs STT, evaluation en conditions realistes.

### Distribution des erreurs STT

| Intensite | Pourcentage | Erreurs appliquees |
|-----------|-------------|-------------------|
| **Clean** | 20% | Aucune erreur |
| **Light** | 40% | 1-2 types d'erreurs |
| **Moderate** | 30% | 2-4 types d'erreurs |
| **Heavy** | 10% | 4+ types d'erreurs |

**Note:** Les entrees TRIP n'ont jamais de troncature (`incomplete=0`) pour preserver les noms de gares dans la phrase.

---

## Types d'erreurs STT simulees

Le module `src/data/stt_augmentation.py` simule **14 types d'erreurs** basees sur les comportements reels de Whisper :

| # | Type d'erreur | Probabilite | Exemples |
|---|---------------|-------------|----------|
| 1 | **Mots de remplissage** | 15% | "euh", "hum", "ben", "alors", "bah", "enfin", "quoi" |
| 2 | **Faux departs** | 8% | "Je veux... enfin je voudrais", "Je... je voudrais" |
| 3 | **Repetitions** | 5% | "Je je voudrais", "de de Paris" |
| 4 | **Phrases incompletes** | 3% | "Je veux aller de Paris a...", "Un train de..." |
| 5 | **Confusions phonetiques** | 10% | "trin" (train), "voyaje" (voyage), "bilet" (billet) |
| 6 | **Erreurs sur noms de gares** | 12% | "Monparnasse", "Lion" (Lyon), "Marseye" (Marseille) |
| 7 | **Erreurs de ponctuation** | 20% | Ponctuation manquante ou erronee |
| 8 | **Erreurs de capitalisation** | 25% | Tout en minuscules, majuscules aleatoires |
| 9 | **Format des nombres** | 8% | "8 heures" vs "huit heures" |
| 10 | **Code-switching** | 2% | "I want aller a Paris", "Je want to go Lyon" |
| 11 | **Hallucinations Whisper** | 1.5% | "Merci d'avoir regarde cette video", "N'oubliez pas de vous abonner" |
| 12 | **Artefacts de bruit** | 5% | "[inaudible]", "[bruit]", "[musique]", "..." |
| 13 | **Variations d'accent** | 6% | Transcriptions phonetiques regionales |
| 14 | **Homophones** | 8% | "a/a", "ou/ou", "vers/vert", "et/est" |

---

## Structure commune

### Colonnes

| Colonne | Type | Description |
|---------|------|-------------|
| `sentence_id` | string | Identifiant unique (BASE/STT + 6 chiffres) |
| `sentence` | string | Texte de la phrase |
| `intent` | string | TRIP, NOT_TRIP, UNKNOWN |
| `language` | string | FRENCH, ENGLISH, SPANISH, GERMAN, ITALIAN, UNKNOWN |
| `departure` | string | Gare de depart (nom propre, vide si non applicable) |
| `destination` | string | Gare d'arrivee (nom propre, vide si non applicable) |
| `intermediate` | string | Gare intermediaire "via" (nom propre, vide si non applicable) |

### Separation Intent / Language

**Important:** L'intention et la langue sont des concepts separes :
- Une phrase en anglais demandant un trajet a `intent=TRIP` et `language=ENGLISH`
- Une phrase en francais non liee au voyage a `intent=NOT_TRIP` et `language=FRENCH`
- Les phrases incomprehensibles ont `intent=UNKNOWN` et `language=UNKNOWN`

---

## Distribution des donnees

> **Note:** Distribution optimisee pour un systeme ferroviaire francais realiste (v0.3.8).

### Distribution par intention

| Intention | Nombre | Pourcentage | Description |
|-----------|--------|-------------|-------------|
| **TRIP** | 74 000 | 74% | Demandes de voyage (toutes langues) |
| **NOT_TRIP** | 25 000 | 25% | Phrases non liees au voyage |
| **UNKNOWN** | 1 000 | 1% | Bruit, gibberish, incomprehensible |

### Distribution des arrets intermediaires (TRIP)

| Type | Nombre | Pourcentage |
|------|--------|-------------|
| **Sans intermediaire** | ~63 000 | 85% |
| **Avec intermediaire** | ~11 000 | 15% |

### Distribution par langue

| Langue | Nombre | Pourcentage | Rationale |
|--------|--------|-------------|-----------|
| **FRENCH** | 89 100 | 89.1% | Langue principale du service SNCF |
| **ENGLISH** | 6 440 | 6.4% | Touristes, lingua franca |
| **SPANISH** | 865 | 0.9% | Usage minoritaire |
| **GERMAN** | 865 | 0.9% | Usage minoritaire |
| **ITALIAN** | 620 | 0.6% | Usage minoritaire |
| **UNKNOWN** | 2 110 | 2.1% | Mixed language + noise |

### Distribution croisee (Intention x Langue)

| Intention | FRENCH | ENGLISH | SPANISH | GERMAN | ITALIAN | UNKNOWN |
|-----------|--------|---------|---------|--------|---------|---------|
| TRIP | 66 600 | 4 440 | 740 | 740 | 370 | 1 110 |
| NOT_TRIP | 22 500 | 2 000 | 125 | 125 | 250 | - |
| UNKNOWN | - | - | - | - | - | 1 000 |

### Caracteristiques des phrases

| Metrique | Valeur | Description |
|----------|--------|-------------|
| **TRIP < 5 mots** | ~14% | Phrases courtes (ultra-minimal, destination-only) |
| **TRIP ultra-minimal** | ~10% | Patterns "Paris Lyon", "Vers Lyon" (TRIP_TEMPLATES_FR_MINIMAL) |
| **NOT_TRIP < 5 mots** | ~82% | Salutations, questions courtes (realiste) |
| **Avec fillers** | ~12% | "euh", "ben", "voila", etc. (naturel STT) |

---

## Scripts utilises

### Script de generation (dataset base)

**Chemin:** `datasets/scripts/generate_base.py`

```bash
poetry run python datasets/scripts/generate_base.py \
    --count 100000 \
    --seed 42 \
    --output datasets/base/
```

**Options:**
- `--count`: Nombre total d'entrees (defaut: 100000)
- `--seed`: Graine aleatoire pour reproductibilite (defaut: 42)
- `--output`: Repertoire de sortie (defaut: datasets/base/)
- `--format`: Format de sortie - csv, json, both (defaut: both)

### Script d'augmentation STT

**Chemin:** `datasets/scripts/augment_stt.py`

```bash
poetry run python datasets/scripts/augment_stt.py \
    --input datasets/base/ \
    --output datasets/augmented/ \
    --seed 42
```

**Options:**
- `--input`: Repertoire d'entree (defaut: datasets/base/)
- `--output`: Repertoire de sortie (defaut: datasets/augmented/)
- `--seed`: Graine aleatoire (defaut: 42)
- `--format`: Format - csv, json, both (defaut: both)

### Module d'augmentation STT

**Chemin:** `src/data/stt_augmentation.py`

Classes principales :
- `STTErrorConfig` : Configuration des probabilites d'erreurs (14 parametres)
- `STTAugmenter` : Application des erreurs STT aux textes
- `ERROR_PROFILES` : Profils d'intensite predefinis (clean, light, moderate, heavy)

---

## Exemples

### Dataset Base - TRIP Francais

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
BASE017809,Quand part le prochain train de Viry-Noureuil a Montchanin,TRIP,FRENCH,Viry-Noureuil,Montchanin,
BASE045969,De Aubigny-en-Artois vers Harfleur,TRIP,FRENCH,Aubigny-en-Artois,Harfleur,
```

### Dataset Augmented - Meme entree avec erreurs STT

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
STT017809,quAnd paRt le proChaiN traiN de viry-nouReuil a montchAnin,TRIP,FRENCH,Viry-Noureuil,Montchanin,
STT045969,Ben euh Aubigny-en-Artois vers Harfleur,TRIP,FRENCH,Aubigny-en-Artois,Harfleur,
```

### TRIP - Avec arret intermediaire

**Base:**
```csv
BASE023456,De Paris a Marseille en passant par Lyon,TRIP,FRENCH,Paris,Marseille,Lyon
```

**Augmented:**
```csv
STT023456,euh de paris a marseille en passant par lion,TRIP,FRENCH,Paris,Marseille,Lyon
```

### NOT_TRIP

```csv
BASE092928,Good evening,NOT_TRIP,ENGLISH,,,
BASE082592,J'ai mange a Tours hier,NOT_TRIP,FRENCH,,,
```

### UNKNOWN

```csv
BASE098624,puis... [coupure],UNKNOWN,UNKNOWN,,,
BASE098842,Rendez-vous sur notre site,UNKNOWN,UNKNOWN,,,
```

---

## Limites et considerations

### Limites des datasets

1. **Gares synthetiques** : Les combinaisons depart/destination sont generees aleatoirement et ne refletent pas necessairement des trajets reels.

2. **Pas d'audio source** : Ces datasets simulent des transcriptions, pas de vraies donnees audio.

3. **Erreurs STT simplifiees** : Les erreurs simulees sont basees sur des patterns connus de Whisper mais ne couvrent pas toutes les subtilites.

4. **Distribution uniforme des erreurs** : En realite, les erreurs dependent de la qualite audio, du bruit ambiant, de l'accent.

5. **Langues simplifiees** : Les templates non-francais sont moins varies.

6. **Ground truth preservee** : Les colonnes `departure`, `destination`, `intermediate` contiennent toujours les noms propres, meme si la phrase est corrompue.

### Considerations d'utilisation

- **Entrainement** : Utiliser `train.csv` (70%)
- **Validation hyperparametres** : Utiliser `val.csv` (15%)
- **Evaluation finale** : Utiliser `test.csv` (15%)

- **Choix du dataset** :
  - `base/` pour etablir une baseline sur donnees propres
  - `augmented/` pour entrainer/evaluer en conditions realistes

- **Metriques recommandees** :
  - Intent classification : Accuracy, F1-score
  - Entity extraction : Exact match, Fuzzy match, Precision/Recall/F1

---

## Regeneration des datasets

### Generer le dataset base

```bash
poetry run python datasets/scripts/generate_base.py \
    --count 100000 \
    --seed 42 \
    --output datasets/base/
```

### Generer le dataset augmente

```bash
poetry run python datasets/scripts/augment_stt.py \
    --input datasets/base/ \
    --output datasets/augmented/ \
    --seed 42
```

### Generer un dataset plus petit (pour tests)

```bash
poetry run python datasets/scripts/generate_base.py \
    --count 10000 \
    --seed 42 \
    --output datasets/test_small/
```

---

## Dependances

- `faker` (^22.0) - Generation de donnees francaises
- `unidecode` (^1.3) - Normalisation des accents
- `pandas` (^2.1) - Manipulation de donnees

---

*Datasets generes pour le projet Travel Order Resolver - Epitech T9-AIA*
