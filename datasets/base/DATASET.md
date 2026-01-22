# Dataset Base 100k - Documentation

> Dataset synthetique de demandes de voyage ferroviaire en langage naturel. Version **propre** (sans erreurs STT).

**Date de generation:** Janvier 2025
**Nombre total d'entrees:** 100 000
**Seed de reproductibilite:** 42

---

## Table des matieres

1. [Objectif](#objectif)
2. [Architecture des datasets](#architecture-des-datasets)
3. [Structure du dataset](#structure-du-dataset)
4. [Fichiers generes](#fichiers-generes)
5. [Distribution des donnees](#distribution-des-donnees)
6. [Scripts utilises](#scripts-utilises)
7. [Exemples](#exemples)
8. [Limites et considerations](#limites-et-considerations)
9. [Regeneration du dataset](#regeneration-du-dataset)

---

## Objectif

Ce dataset a ete cree pour entrainer et evaluer des modeles NLP capables de :

1. **Classifier l'intention** (TRIP, NOT_TRIP, UNKNOWN)
2. **Detecter la langue** (FRENCH, ENGLISH, SPANISH, GERMAN, ITALIAN, UNKNOWN)
3. **Extraire les entites** (gare de depart, gare d'arrivee, gare intermediaire)

Ce dataset **base** contient des phrases propres, sans erreurs de transcription. Pour simuler des erreurs speech-to-text realistes, utilisez le script d'augmentation (voir section [Architecture des datasets](#architecture-des-datasets)).

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

## Structure du dataset

### Colonnes

| Colonne | Type | Description |
|---------|------|-------------|
| `sentence_id` | string | Identifiant unique (format: BASE000001) |
| `sentence` | string | Texte propre (sans erreurs STT) |
| `intent` | string | Intention detectee: TRIP, NOT_TRIP, UNKNOWN |
| `language` | string | Langue detectee: FRENCH, ENGLISH, SPANISH, GERMAN, ITALIAN, UNKNOWN |
| `departure` | string | Gare de depart (nom propre, vide si non applicable) |
| `destination` | string | Gare d'arrivee (nom propre, vide si non applicable) |
| `intermediate` | string | Gare intermediaire "via" (nom propre, vide si non applicable) |

### Separation Intent / Language

**Important:** L'intention et la langue sont des concepts separes :
- Une phrase en anglais demandant un trajet a `intent=TRIP` et `language=ENGLISH`
- Une phrase en francais non liee au voyage a `intent=NOT_TRIP` et `language=FRENCH`
- Les phrases incomprehensibles ont `intent=UNKNOWN` et `language=UNKNOWN`

---

## Fichiers generes

| Fichier | Taille | Lignes | Description |
|---------|--------|--------|-------------|
| `base_dataset_100k.csv` | ~7 MB | 100 000 | Dataset complet |
| `base_dataset_100k.json` | ~18 MB | 100 000 | Dataset complet (format JSON) |
| `train.csv` | ~5 MB | 69 998 | Ensemble d'entrainement (70%) |
| `train.json` | ~13 MB | 69 998 | Ensemble d'entrainement (JSON) |
| `val.csv` | ~1 MB | 14 997 | Ensemble de validation (15%) |
| `val.json` | ~3 MB | 14 997 | Ensemble de validation (JSON) |
| `test.csv` | ~1 MB | 15 005 | Ensemble de test (15%) |
| `test.json` | ~3 MB | 15 005 | Ensemble de test (JSON) |

Le split est **stratifie** : chaque ensemble conserve la meme distribution d'intentions et de langues.

---

## Distribution des donnees

### Distribution par intention

| Intention | Nombre | Pourcentage | Description |
|-----------|--------|-------------|-------------|
| **TRIP** | 70 000 | 70% | Demandes de voyage (toutes langues) |
| **NOT_TRIP** | 25 000 | 25% | Phrases non liees au voyage |
| **UNKNOWN** | 5 000 | 5% | Bruit, hallucinations, incomprehensible |

### Distribution des arrets intermediaires (TRIP)

Parmi les 70 000 entrees TRIP, environ **15%** (~10 500) incluent un arret intermediaire ("via", "en passant par").

| Type | Nombre | Pourcentage |
|------|--------|-------------|
| **Sans intermediaire** | ~59 500 | 85% |
| **Avec intermediaire** | ~10 500 | 15% |

Les arrets intermediaires sont disponibles pour toutes les langues (sauf UNKNOWN/mixed).

### Distribution par langue

| Langue | Nombre | Pourcentage |
|--------|--------|-------------|
| **FRENCH** | 76 000 | 76% |
| **ENGLISH** | 10 000 | 10% |
| **SPANISH** | 3 249 | 3.2% |
| **GERMAN** | 3 249 | 3.2% |
| **ITALIAN** | 1 501 | 1.5% |
| **UNKNOWN** | 6 001 | 6% |

### Distribution croisee (Intention x Langue)

| Intention | FRENCH | ENGLISH | SPANISH | GERMAN | ITALIAN | UNKNOWN |
|-----------|--------|---------|---------|--------|---------|---------|
| TRIP | 56 000 | 7 000 | 2 499 | 2 499 | 1 001 | 1 001 |
| NOT_TRIP | 20 000 | 3 000 | 750 | 750 | 500 | - |
| UNKNOWN | - | - | - | - | - | 5 000 |

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

Ce script lit les fichiers du dataset base et applique des erreurs STT realistes (mots de remplissage, erreurs de capitalisation, confusions phonetiques, etc.). Les IDs sont changes de `BASE` a `STT`.

### Module d'augmentation STT

**Chemin:** `src/data/stt_augmentation.py`

Classes principales :
- `STTErrorConfig` : Configuration des probabilites d'erreurs
- `STTAugmenter` : Application des erreurs STT aux textes
- `ERROR_PROFILES` : Profils d'intensite predefinis (clean, light, moderate, heavy)

---

## Exemples

### TRIP - Francais

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
BASE017809,Quand part le prochain train de Viry-Noureuil a Montchanin,TRIP,FRENCH,Viry-Noureuil,Montchanin,
BASE045969,De Aubigny-en-Artois vers Harfleur,TRIP,FRENCH,Aubigny-en-Artois,Harfleur,
BASE016764,Je file a Saverdun,TRIP,FRENCH,,Saverdun,
```

### TRIP - Avec arret intermediaire

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
BASE023456,De Paris a Marseille en passant par Lyon,TRIP,FRENCH,Paris,Marseille,Lyon
BASE034567,Je voudrais aller de Lille a Nice via Paris,TRIP,FRENCH,Lille,Nice,Paris
BASE045678,From London to Rome via Paris,TRIP,ENGLISH,London,Rome,Paris
BASE056789,Bordeaux puis Toulouse puis Montpellier,TRIP,FRENCH,Bordeaux,Montpellier,Toulouse
```

### TRIP - Autres langues

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
BASE061595,Ticket from L'Ariane La Trinite to Boisleux-au-Mont please,TRIP,ENGLISH,L'Ariane La Trinite,Boisleux-au-Mont,
BASE067176,Ein Ticket von Vaas nach Angouleme,TRIP,GERMAN,Vaas,Angouleme,
```

### NOT_TRIP

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
BASE092928,Good evening,NOT_TRIP,ENGLISH,,,
BASE082592,J'ai mange a Tours hier,NOT_TRIP,FRENCH,,,
BASE077440,Tu es qui,NOT_TRIP,FRENCH,,,
```

### UNKNOWN

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
BASE098624,puis... [coupure],UNKNOWN,UNKNOWN,,,
BASE098842,Rendez-vous sur notre site,UNKNOWN,UNKNOWN,,,
BASE099087,a,UNKNOWN,UNKNOWN,,,
```

---

## Limites et considerations

### Limites du dataset

1. **Gares synthetiques** : Les combinaisons depart/destination sont generees aleatoirement et ne refletent pas necessairement des trajets reels ou frequents.

2. **Pas d'audio source** : Ce dataset ne contient pas de fichiers audio - il simule uniquement des demandes textuelles.

3. **Langues simplifiees** : Les templates non-francais sont moins varies et ne representent pas la diversite reelle des accents et formulations.

4. **Templates repetitifs** : Les memes structures de phrases peuvent apparaitre dans train/val/test avec des gares differentes.

### Considerations d'utilisation

- **Entrainement** : Utiliser `train.csv` (70%)
- **Validation hyperparametres** : Utiliser `val.csv` (15%)
- **Evaluation finale** : Utiliser `test.csv` (15%)

- **Attention au leakage** : Les memes templates peuvent apparaitre dans train/val/test avec des gares differentes. Pour une evaluation stricte, considerer un split par template.

- **Metriques recommandees** :
  - Intent classification : Accuracy, F1-score
  - Entity extraction : Exact match, Fuzzy match, Precision/Recall/F1

---

## Regeneration du dataset

Pour regenerer le dataset base :

```bash
cd /root/code/Romain-Ber/Epitech/T9-AIA/TravelOrderResolver
poetry run python datasets/scripts/generate_base.py \
    --count 100000 \
    --seed 42 \
    --output datasets/base/
```

Pour generer ensuite la version augmentee (avec erreurs STT) :

```bash
poetry run python datasets/scripts/augment_stt.py \
    --input datasets/base/ \
    --output datasets/augmented/ \
    --seed 42
```

Pour generer un dataset plus petit (pour tests) :

```bash
poetry run python datasets/scripts/generate_base.py \
    --count 10000 \
    --seed 42 \
    --output datasets/test_small/
```

---

## Dependances

Le script utilise les bibliotheques suivantes (deja installees via Poetry) :

- `faker` (^22.0) - Generation de donnees francaises
- `unidecode` (^1.3) - Normalisation des accents
- `pandas` (^2.1) - Manipulation de donnees

---

*Dataset genere pour le projet Travel Order Resolver - Epitech T9-AIA*
