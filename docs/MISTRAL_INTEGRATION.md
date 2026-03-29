# Intégration Ministral 3B — Plan

> **Source** : branche `origin/feat/mistral-qlora` (Romain)
> **Date audit** : 28 mars 2026
> **Objectif** : Ajouter un 5e backend NLP (Ministral 3B) en zero-shot ET fine-tuné,
> pour comparer académiquement l'impact du fine-tuning sur un LLM génératif.

---

## 1. Ce qui existe sur la branche

| Fichier | Rôle | Status |
|---------|------|--------|
| `src/nlp/entity/ministral_entity.py` | EntityExtractor génératif (output JSON) | ✅ Complet |
| `src/nlp/intent/ministral_intent.py` | IntentClassifier génératif | ⚠️ Bug : `adapter_path` dupliqué l.41-42 |
| `src/nlp/unified/ministral_unified.py` | 3 interfaces en 1 pass (unsloth) | ❌ CUDA only, pas pour la démo |
| `src/utils/quantization.py` | Utilitaires QLoRA | ✅ Complet |
| `training/config/ministral_entity.yaml` | Config fine-tuning entity | ✅ Prêt |
| `training/config/ministral_intent.yaml` | Config fine-tuning intent | ✅ Prêt |
| `training/notebooks/ministral_unified_colab.ipynb` | Notebook Colab | ✅ Prêt à exécuter |
| `training/scripts/train_ministral_*.py` | Scripts d'entraînement | ✅ Complet |
| `unsloth_compiled_cache/` | Cache compilé (~70 fichiers) | ❌ NE PAS MERGER |

---

## 2. Bugs à corriger

### Bug 1 : `adapter_path` dupliqué (SyntaxError)

Dans `src/nlp/intent/ministral_intent.py`, ligne 41-42 :
```python
# AVANT (cassé)
def __init__(
    self,
    model_name: str = "unsloth/Ministral-3-3B-Base-2512-bnb-4bit",
    adapter_path: Optional[str] = None,
    adapter_path: Optional[str] = None,  # ← DOUBLON À SUPPRIMER
    device: DeviceType = "auto",
```

---

## 3. Plan d'intégration

### Étape 1 — Zero-shot (sans fine-tuning)

Objectif : montrer les performances brutes d'un LLM 3B non fine-tuné.

1. Cherry-pick les fichiers utiles (PAS tout merger) :
   - `src/nlp/entity/ministral_entity.py`
   - `src/nlp/intent/ministral_intent.py` (avec fix bug)
   - `src/utils/quantization.py`
2. Ajouter au `model_factory.py` :
   ```python
   if "ministral" in models or "all" in models:
       from src.nlp.entity import MinistralEntityExtractor
       extractors.append(("Ministral", MinistralEntityExtractor()))
   ```
3. Ajouter aux `__init__.py` des modules entity et intent
4. Le modèle de base (`unsloth/Ministral-3-3B-Base-2512-bnb-4bit`) sera
   téléchargé automatiquement depuis HuggingFace (~2 GB)
5. Lancer l'évaluation → obtenir les métriques zero-shot

**Résultat attendu** : performances faibles (le modèle n'est pas entraîné
pour notre tâche), ce qui justifie académiquement le fine-tuning.

### Étape 2 — Fine-tuning sur Google Colab

Objectif : fine-tuner le modèle et comparer avant/après.

1. Ouvrir `training/notebooks/ministral_unified_colab.ipynb` sur Colab
2. Uploader `datasets/base/train.json` et `datasets/base/val.json`
3. Exécuter le notebook (~1h sur GPU T4 gratuit)
4. Télécharger les poids LoRA résultants
5. Placer dans `models/ministral-entity-lora/` et `models/ministral-intent-lora/`
6. Mettre à jour l'instanciation avec `adapter_path` :
   ```python
   MinistralEntityExtractor(adapter_path="models/ministral-entity-lora")
   ```
7. Relancer l'évaluation → comparer zero-shot vs fine-tuné

### Étape 3 — Analyse comparative

Tableau final dans le rapport :

| Modèle | Type | Entity F1 | Intent Acc | Latence |
|--------|------|-----------|------------|---------|
| Regex | Règles | ... | ... | ... |
| SpaCy | NER générique | ... | ... | ... |
| CamemBERT | Fine-tuné NER (BIO) | ... | ... | ... |
| Flan-T5 | Fine-tuné seq2seq | ... | ... | ... |
| Ministral (zero-shot) | LLM 3B génératif | ... | ... | ... |
| Ministral (fine-tuné) | LLM 3B + QLoRA | ... | ... | ... |

**Angle académique** : progression Regex → NER générique → Fine-tuning classique → LLM génératif, et impact du fine-tuning sur un LLM.

---

## 4. Dépendances à ajouter

Pour le zero-shot (entity/intent séparés, compatible CPU) :
```
peft >= 0.7
bitsandbytes >= 0.41  # optionnel, uniquement pour 4-bit GPU
```

Pour le fine-tuning (Colab uniquement) :
```
unsloth
trl
```

---

## 5. Ce qu'il ne faut PAS merger

- `unsloth_compiled_cache/` — artefact de compilation locale (~70 fichiers)
- `src/nlp/unified/ministral_unified.py` — dépend d'unsloth (CUDA only)
- Les modifications au `Makefile` et `pyproject.toml` de la branche (potentiels conflits)

---

## 6. Commandes pour l'intégration

```bash
# Cherry-pick les fichiers depuis la branche (sans checkout)
git show origin/feat/mistral-qlora:src/nlp/entity/ministral_entity.py > src/nlp/entity/ministral_entity.py
git show origin/feat/mistral-qlora:src/nlp/intent/ministral_intent.py > src/nlp/intent/ministral_intent.py
git show origin/feat/mistral-qlora:src/utils/quantization.py > src/utils/quantization.py
git show origin/feat/mistral-qlora:training/config/ministral_entity.yaml > training/config/ministral_entity.yaml
git show origin/feat/mistral-qlora:training/config/ministral_intent.yaml > training/config/ministral_intent.yaml

# Puis fixer le bug adapter_path dupliqué dans ministral_intent.py
# Puis ajouter au model_factory.py et aux __init__.py
```
