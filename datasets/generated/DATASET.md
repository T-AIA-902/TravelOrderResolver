# Dataset STT 100k - Documentation

> Dataset synthetique simulant des sorties de transcription speech-to-text (Whisper) pour des demandes de voyage ferroviaire.

**Date de generation:** Janvier 2025
**Nombre total d'entrees:** 100 000
**Seed de reproductibilite:** 42

---

## Table des matieres

1. [Objectif](#objectif)
2. [Structure du dataset](#structure-du-dataset)
3. [Fichiers generes](#fichiers-generes)
4. [Distribution des donnees](#distribution-des-donnees)
5. [Types d'erreurs STT simulees](#types-derreurs-stt-simulees)
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

Le dataset simule des transcriptions speech-to-text realistes avec les erreurs typiques de Whisper :
- Mots de remplissage et hesitations
- Erreurs de capitalisation et ponctuation
- Confusions phonetiques
- Hallucinations
- Phrases tronquees

---

## Structure du dataset

### Colonnes

| Colonne | Type | Description |
|---------|------|-------------|
| `sentence_id` | string | Identifiant unique (format: STT000001) |
| `sentence` | string | Texte transcrit (peut contenir des erreurs STT) |
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
| `stt_dataset_100k.csv` | 7.9 MB | 100 000 | Dataset complet |
| `stt_dataset_100k.json` | 20 MB | 100 000 | Dataset complet (format JSON) |
| `train.csv` | 5.5 MB | 69 998 | Ensemble d'entrainement (70%) |
| `train.json` | 14 MB | 69 998 | Ensemble d'entrainement (JSON) |
| `val.csv` | 1.2 MB | 14 997 | Ensemble de validation (15%) |
| `val.json` | 3.0 MB | 14 997 | Ensemble de validation (JSON) |
| `test.csv` | 1.2 MB | 15 005 | Ensemble de test (15%) |
| `test.json` | 3.0 MB | 15 005 | Ensemble de test (JSON) |

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

### Distribution des erreurs STT (pour TRIP francais)

| Intensite | Pourcentage | Erreurs appliquees |
|-----------|-------------|-------------------|
| **Clean** | 20% | Aucune erreur |
| **Light** | 40% | 1-2 types d'erreurs |
| **Moderate** | 30% | 2-4 types d'erreurs |
| **Heavy** | 10% | 4+ types d'erreurs |

---

## Types d'erreurs STT simulees

Le module `stt_augmentation.py` simule 14 types d'erreurs basees sur les comportements reels de Whisper :

| Type d'erreur | Probabilite | Exemples |
|---------------|-------------|----------|
| **Mots de remplissage** | 15% | "euh", "hum", "ben", "alors", "bah", "enfin", "quoi" |
| **Faux departs** | 8% | "Je veux... enfin je voudrais", "Je... je voudrais" |
| **Repetitions** | 5% | "Je je voudrais", "de de Paris" |
| **Phrases incompletes** | 3% | "Je veux aller de Paris a...", "Un train de..." |
| **Confusions phonetiques** | 10% | "trin" (train), "voyaje" (voyage), "bilet" (billet) |
| **Erreurs sur noms de gares** | 12% | "Monparnasse", "Lion" (Lyon), "Marseye" (Marseille) |
| **Erreurs de ponctuation** | 20% | Ponctuation manquante ou erronee |
| **Erreurs de capitalisation** | 25% | Tout en minuscules, majuscules aleatoires |
| **Format des nombres** | 8% | "8 heures" vs "huit heures" |
| **Code-switching** | 2% | "I want aller a Paris", "Je want to go Lyon" |
| **Hallucinations Whisper** | 1.5% | "Merci d'avoir regarde cette video", "N'oubliez pas de vous abonner" |
| **Artefacts de bruit** | 5% | "[inaudible]", "[bruit]", "[musique]", "..." |
| **Variations d'accent** | 6% | Transcriptions phonetiques regionales |
| **Homophones** | 8% | "a/a", "ou/ou", "vers/vert", "et/est" |

---

## Scripts utilises

### Script de generation principal

**Chemin:** `datasets/scripts/generate_stt_dataset.py`

```bash
poetry run python datasets/scripts/generate_stt_dataset.py \
    --count 100000 \
    --seed 42 \
    --output datasets/generated/
```

**Options:**
- `--count`: Nombre total d'entrees (defaut: 100000)
- `--seed`: Graine aleatoire pour reproductibilite (defaut: 42)
- `--output`: Repertoire de sortie (defaut: datasets/generated/)
- `--format`: Format de sortie - csv, json, both (defaut: both)

### Module d'augmentation STT

**Chemin:** `src/data/stt_augmentation.py`

Classes principales :
- `STTErrorConfig` : Configuration des probabilites d'erreurs
- `STTAugmenter` : Application des erreurs STT aux textes
- `create_augmenter(intensity)` : Factory pour creer un augmenter avec un profil predefini

Profils d'intensite disponibles :
- `"clean"` : Aucune erreur
- `"light"` : Erreurs legeres
- `"moderate"` : Erreurs moderees
- `"heavy"` : Erreurs importantes

### Script original de reference

**Chemin:** `datasets/scripts/generate_sentences.py`

Script original avec ~67 templates. Le nouveau script `generate_stt_dataset.py` etend ce travail avec 200+ templates par categorie.

---

## Exemples

### TRIP - Francais avec erreurs STT

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
STT017809,quAnd paRt le proChaiN traiN de viry-nouReuil a montchAnin,TRIP,FRENCH,Viry-Noureuil,Montchanin,
STT045969,Ben euh Aubigny-en-Artois vers Harfleur,TRIP,FRENCH,Aubigny-en-Artois,Harfleur,
STT016764,Je hein file euh a ouais Saverdun,TRIP,FRENCH,,Saverdun,
```

### TRIP - Avec arret intermediaire

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
STT023456,De Paris a Marseille en passant par Lyon,TRIP,FRENCH,Paris,Marseille,Lyon
STT034567,Euh je voudrais aller de Lille a Nice via Paris,TRIP,FRENCH,Lille,Nice,Paris
STT045678,From London to Rome via Paris,TRIP,ENGLISH,London,Rome,Paris
STT056789,Bordeaux puis Toulouse puis Montpellier,TRIP,FRENCH,Bordeaux,Montpellier,Toulouse
```

### TRIP - Autres langues

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
STT061595,Ticket from L'Ariane La Trinité to Boisleux-au-Mont please,TRIP,ENGLISH,L'Ariane La Trinité,Boisleux-au-Mont,
STT067176,Ein Ticket von Vaas nach Angoulême,TRIP,GERMAN,Vaas,Angoulême,
```

### NOT_TRIP

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
STT092928,Good evening,NOT_TRIP,ENGLISH,,,
STT082592,J'ai mange a Tours hier,NOT_TRIP,FRENCH,,,
STT077440,Tu es qui,NOT_TRIP,FRENCH,,,
```

### UNKNOWN

```csv
sentence_id,sentence,intent,language,departure,destination,intermediate
STT098624,puis... [coupure],UNKNOWN,UNKNOWN,,,
STT098842,Rendez-vous sur notre site,UNKNOWN,UNKNOWN,,,
STT099087,a,UNKNOWN,UNKNOWN,,,
```

---

## Limites et considerations

### Limites du dataset

1. **Gares synthetiques** : Les combinaisons depart/destination sont generees aleatoirement et ne refletent pas necessairement des trajets reels ou frequents.

2. **Erreurs STT simplifiees** : Les erreurs simulees sont basees sur des patterns connus de Whisper mais ne couvrent pas toutes les subtilites d'une vraie transcription audio.

3. **Pas d'audio source** : Ce dataset ne contient pas de fichiers audio - il simule uniquement la sortie texte d'un systeme STT.

4. **Distribution des erreurs uniforme** : Les erreurs sont appliquees uniformement, alors qu'en realite elles dependent de la qualite audio, du bruit ambiant, de l'accent du locuteur, etc.

5. **Langues simplifiees** : Les templates non-francais sont moins varies et ne representent pas la diversite reelle des accents et formulations.

6. **Noms de gares preserves** : Les colonnes `departure`, `destination` et `intermediate` contiennent toujours les noms propres des gares, meme si la phrase contient des erreurs. Cela peut creer un decalage pour l'entrainement NER.

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

Pour regenerer le dataset avec les memes parametres :

```bash
cd /root/code/Romain-Ber/Epitech/T9-AIA/TravelOrderResolver
poetry run python datasets/scripts/generate_stt_dataset.py \
    --count 100000 \
    --seed 42 \
    --output datasets/generated/
```

Pour generer un dataset plus petit (pour tests) :

```bash
poetry run python datasets/scripts/generate_stt_dataset.py \
    --count 10000 \
    --seed 42 \
    --output datasets/test_small/
```

Pour changer la distribution, modifier les constantes dans `generate_stt_dataset.py` :
- Lignes ~800-820 : ratios par langue pour TRIP
- Lignes ~825-830 : ratios par langue pour NOT_TRIP

---

## Dependances

Le script utilise les bibliotheques suivantes (deja installees via Poetry) :

- `faker` (^22.0) - Generation de donnees francaises
- `unidecode` (^1.3) - Normalisation des accents
- `pandas` (^2.1) - Manipulation de donnees

---

*Dataset genere pour le projet Travel Order Resolver - Epitech T9-AIA*
