# CamemBERT NER Natif - Modele Re-entraine

## 1. Objectif

Entrainer un modele **CamemBERT natif** (encoder-only) pour la tache de **NER** (Named Entity Recognition) appliquee aux commandes de trajet ferroviaire en francais.

Le modele doit :
- **Identifier** les gares de depart et d'arrivee dans une phrase en langage naturel
- **Rejeter** les phrases non pertinentes avec un code d'erreur

### Sorties attendues

| Cas | Format de sortie |
|-----|-----------------|
| Phrase valide (voyage) | `sentenceID, Departure, Destination` |
| Phrase non francaise | `sentenceID, NOT_FRENCH` |
| Phrase hors domaine | `sentenceID, NOT_TRIP` |
| Cas ambigu | `sentenceID, UNKNOWN` |

### Exemples

```
"Je veux aller de Paris a Lyon"           -> S001, Paris, Lyon
"Reserve-moi un billet Marseille-Nice"    -> S002, Marseille, Nice
"Quel temps fait-il demain ?"             -> S003, NOT_TRIP
"Hello how are you?"                      -> S004, NOT_FRENCH
```

---

## 2. Architecture

### Type : Encodeur-only (BERT) + Token Classification

```
Phrase en entree
      |
      v
[ CamemBERT-base ]         <- Modele pre-entraine HuggingFace (camembert-base)
      |                        768 dimensions, 12 couches, 110M params
      v
[ Classification Head ]    <- Couche lineaire ajoutee (768 -> 5 classes)
      |
      v
[ Labels NER par token ]   <- O | B-DEP | I-DEP | B-DEST | I-DEST
      |
      v
[ Post-processing ]        <- Extraction entites + fuzzy matching + validation
      |
      v
Sortie structuree
```

### Labels NER

| Label | Signification |
|-------|--------------|
| `O` | Token hors entite |
| `B-DEP` | Debut du nom de gare de depart |
| `I-DEP` | Suite du nom de gare de depart |
| `B-DEST` | Debut du nom de gare de destination |
| `I-DEST` | Suite du nom de gare de destination |

### Post-processing

Le modele NER produit des labels par token. Le post-processing :

1. **Regroupe** les tokens B-/I- consecutifs en entites completes
2. **Fuzzy matching** contre la base de gares SNCF (RapidFuzz, seuil 80%)
3. **Detection de langue** (indicateurs lexicaux francais)
4. **Classification** : TRIP valide / NOT_TRIP / NOT_FRENCH / UNKNOWN

---

## 3. Donnees d'entrainement

### Sources (datasets/raw/)

| Fichier | Description | Utilisation |
|---------|-------------|-------------|
| `sentences/travel_sentences.csv` | 12 phrases annotees manuellement (IOB) | Donnees seed, labels B-LOC convertis en B-DEP/B-DEST |
| `sncf/gares.csv` | 17 gares principales + noms de ville | Noms de gares pour la generation |
| `sncf/gares-de-voyageurs.json` | ~3000 gares de voyageurs SNCF | Extension du vocabulaire de gares |

### Generation synthetique

- **50 templates** de phrases de voyage en francais
  - Structures variees : "de X a Y", "depuis X vers Y", "billet pour Y au depart de X"...
- **40 phrases** non-voyage (NOT_TRIP)
- Generation par combinaison template x gare aleatoire

### Dataset genere (defaut)

| Split | Taille | Ratio |
|-------|--------|-------|
| Train | ~5200 | 80% |
| Validation | ~650 | 10% |
| Test | ~650 | 10% |

Stockage : `datasets/processed/ner_retrain/`

---

## 4. Pipeline d'entrainement

### Etape 1 : Preparation des donnees

```bash
python -m training.camembert_ner_native.prepare_data
```

Options :
- `--n_trip 5000` : nombre de phrases TRIP a generer
- `--n_not_trip 1500` : nombre de phrases NOT_TRIP
- `--seed 42` : graine aleatoire

Produit :
- `datasets/processed/ner_retrain/train.json`
- `datasets/processed/ner_retrain/val.json`
- `datasets/processed/ner_retrain/test.json`
- `datasets/processed/ner_retrain/label_config.json`

### Etape 2 : Entrainement

```bash
python -m training.camembert_ner_native.train
```

Options :
- `--epochs 5` : nombre d'epoques
- `--batch_size 16` : taille de batch
- `--lr 2e-5` : learning rate
- `--max_length 128` : longueur max des sequences
- `--warmup_ratio 0.1` : ratio de warmup
- `--weight_decay 0.01` : regularisation

Parametres d'entrainement :
- Modele de base : `camembert-base` (HuggingFace)
- Strategie : evaluation et sauvegarde a chaque epoque
- Early stopping : patience de 3 epoques sur le F1
- Meilleur modele charge en fin d'entrainement
- FP16 active si GPU disponible

Sortie : `models/camembert-ner-retrain/`

### Etape 3 : Evaluation

```bash
python -m training.camembert_ner_native.evaluate
```

Options :
- `--model_dir models/camembert-ner-retrain` : chemin du modele
- `--test_path datasets/processed/ner_retrain/test.json` : donnees de test
- `--show_examples 10` : nombre d'exemples a afficher

Produit :
- Rapport de classification (precision, recall, F1 par entite)
- Matrice de confusion token-level
- Exemples de predictions
- `reports/camembert_ner_retrain_eval.json`

### Etape 4 : Inference

```bash
# Phrase unique
python -m training.camembert_ner_native.inference --sentence "Je veux aller de Paris a Lyon"

# Fichier CSV
python -m training.camembert_ner_native.inference --input_csv data.csv --output_csv results.csv

# Mode interactif
python -m training.camembert_ner_native.inference
```

Options :
- `--no_fuzzy` : desactiver le fuzzy matching
- `--verbose` : afficher le detail NER

---

## 5. Structure des fichiers

```
training/camembert_ner_native/
  __init__.py              # Module init
  prepare_data.py          # Generation du dataset NER
  train.py                 # Entrainement du modele
  evaluate.py              # Evaluation + metriques
  inference.py             # Inference + post-processing

datasets/processed/ner_retrain/
  train.json               # Donnees d'entrainement
  val.json                 # Donnees de validation
  test.json                # Donnees de test
  label_config.json        # Configuration des labels

models/camembert-ner-retrain/
  config.json              # Config du modele
  model.safetensors        # Poids du modele
  tokenizer.json           # Tokenizer CamemBERT
  tokenizer_config.json
  sentencepiece.bpe.model
  special_tokens_map.json
  label_config.json        # Mapping labels NER
```

---

## 6. Dependances

Dependances requises (deja dans `pyproject.toml`) :

| Package | Version | Usage |
|---------|---------|-------|
| `transformers` | ^4.36 | Modele CamemBERT, Trainer |
| `torch` | ^2.1 | Backend deep learning |
| `pandas` | ^2.1 | Lecture des CSV |
| `numpy` | ^1.26 | Calculs numeriques |
| `rapidfuzz` | ^3.5 | Fuzzy matching gares |
| `seqeval` | * | Metriques NER (optionnel) |

Installation de seqeval si necessaire :
```bash
pip install seqeval
```

---

## 7. Differences avec les autres modeles

| | CamemBERT natif (baseline) | CamemBERT NER retrain (ce modele) | CamemBERT fine-tuned |
|---|---|---|---|
| **Approche** | Fuzzy matching + regles | NER token classification | NER token classification |
| **Apprentissage** | Aucun | Depuis camembert-base | Depuis modele NER existant |
| **Labels** | N/A | B-DEP/I-DEP/B-DEST/I-DEST | B-DEP/I-DEP/B-DEST/I-DEST |
| **Post-processing** | Regles linguistiques | Fuzzy matching + validation | Fuzzy matching + validation |
| **Generalisation** | Faible (regles fixes) | Moyenne (donnees synthetiques) | Bonne (donnees variees) |
| **Avantage** | Aucun entrainement | Controle total du pipeline | Meilleures performances |
| **Modele source** | camembert-base | camembert-base | camembert-ner pre-entraine |
| **Emplacement** | `models/camembert-ner/` | `models/camembert-ner-retrain/` | `models/camembert-ner-finetuned/` |

---

## 8. Limites et ameliorations possibles

### Limites actuelles

- **Donnees synthetiques** : les templates de generation couvrent un nombre limite de formulations
- **Pas de gestion des gares intermediaires** : seuls depart et destination sont extraits
- **Detection de langue simplifiee** : basee sur des indicateurs lexicaux, pas un modele dedie
- **Pas d'augmentation STT** : les donnees d'entrainement ne contiennent pas d'erreurs de transcription

### Ameliorations possibles

1. **Enrichir les templates** : ajouter plus de formulations, structures complexes, argot
2. **Ajouter l'augmentation STT** : utiliser `src/data/stt_augmentation.py` pour generer des variantes avec erreurs
3. **Ajouter les gares intermediaires** : etendre les labels avec B-INTER / I-INTER
4. **Utiliser le dataset base 100k** : generer des labels NER a partir des colonnes departure/destination
5. **Data augmentation** : synonymes, paraphrases, variations de ponctuation
6. **Evaluation sur donnees reelles** : tester sur des transcriptions vocales reelles


UTILISATION:

# 1. Preparer les donnees
python -m training.camembert_ner_retrain.prepare_data

# 2. Entrainer le modele (sauvegarde dans models/camembert-ner-retrain/)
python -m models.camembert-ner-retrain.train

# 3. Evaluer
python -m training.camembert_ner_retrain.evaluate

# 4. Inference
python -m training.camembert_ner_retrain.inference --sentence "Je veux aller de Paris a Lyon"


