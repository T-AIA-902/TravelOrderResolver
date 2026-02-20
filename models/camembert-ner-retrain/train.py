"""
Entrainement CamemBERT natif pour NER (Token Classification).

Modele de base : camembert-base (HuggingFace)
Architecture  : CamembertForTokenClassification (encoder-only + linear head)
Labels        : O, B-DEP, I-DEP, B-DEST, I-DEST

Usage:
    python models/camembert-ner-retrain/train.py
    python models/camembert-ner-retrain/train.py --epochs 10 --batch_size 32 --lr 3e-5
    python models/camembert-ner-retrain/train.py --data_dir datasets/augmented
"""

import json
import argparse
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    CamembertTokenizerFast,
    CamembertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "augmented"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "camembert-ner-retrain"

LABEL_LIST = ["O", "B-DEP", "I-DEP", "B-DEST", "I-DEST"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}


# ---------------------------------------------------------------------------
# Conversion augmented -> NER à la volée
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalise le texte pour la comparaison."""
    text = text.lower().strip()
    text = re.sub(r"['\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_entity_span(tokens: list[str], entity: str) -> tuple[int, int] | None:
    """Trouve la position d'une entité dans la liste de tokens."""
    if not entity or not tokens:
        return None

    entity_normalized = normalize_text(entity)
    entity_words = entity_normalized.split()

    if not entity_words:
        return None

    for i in range(len(tokens) - len(entity_words) + 1):
        match = True
        for j, entity_word in enumerate(entity_words):
            token_normalized = normalize_text(tokens[i + j])
            if token_normalized != entity_word:
                match = False
                break
        if match:
            return (i, i + len(entity_words))

    return None


def convert_augmented_sample(sample: dict) -> dict | None:
    """Convertit un échantillon augmented vers le format NER tokenisé."""
    sentence = sample.get("sentence", "")
    departure = sample.get("departure", "").strip()
    destination = sample.get("destination", "").strip()
    intent = sample.get("intent", "TRIP")

    tokens = sentence.split()
    if not tokens:
        return None

    labels = ["O"] * len(tokens)

    if intent == "NOT_TRIP":
        return {"tokens": tokens, "labels": labels}

    entities_found = 0

    if departure:
        span = find_entity_span(tokens, departure)
        if span:
            start, end = span
            labels[start] = "B-DEP"
            for i in range(start + 1, end):
                labels[i] = "I-DEP"
            entities_found += 1

    if destination:
        span = find_entity_span(tokens, destination)
        if span:
            start, end = span
            if labels[start] == "O":
                labels[start] = "B-DEST"
                for i in range(start + 1, end):
                    if labels[i] == "O":
                        labels[i] = "I-DEST"
                entities_found += 1

    if intent == "TRIP" and entities_found == 0 and (departure or destination):
        return None

    return {"tokens": tokens, "labels": labels}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NERDataset(Dataset):
    """Dataset NER avec alignement subword pour CamemBERT."""

    def __init__(self, data: list, tokenizer: CamembertTokenizerFast, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._prepare_data(data)

    def _prepare_data(self, raw_data: list) -> list:
        """Prépare les données (conversion si format augmented)."""
        prepared = []
        for sample in raw_data:
            if "tokens" in sample and "labels" in sample:
                # Format déjà tokenisé
                prepared.append(sample)
            else:
                # Format augmented -> conversion à la volée
                converted = convert_augmented_sample(sample)
                if converted:
                    prepared.append(converted)
        return prepared

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        words = sample["tokens"]
        word_labels = sample["labels"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Alignement des labels sur les subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Tokens speciaux (<s>, </s>, <pad>)
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                # Premier subword du mot -> label du mot
                if word_id < len(word_labels):
                    aligned_labels.append(LABEL2ID.get(word_labels[word_id], 0))
                else:
                    aligned_labels.append(-100)
            else:
                # Subword suivant -> ignore dans le loss
                aligned_labels.append(-100)
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Metriques
# ---------------------------------------------------------------------------
def compute_metrics(eval_preds):
    """Calcul des metriques NER (precision, recall, F1) par entite."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_filtered = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_seq.append(ID2LABEL[label_id])
            pred_seq_filtered.append(ID2LABEL[pred_id])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_filtered)

    # Metriques par entite avec seqeval
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score

        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
    except ImportError:
        # Fallback : accuracy token-level
        correct = 0
        total = 0
        for true_seq, pred_seq in zip(true_labels, pred_labels):
            for t, p in zip(true_seq, pred_seq):
                total += 1
                if t == p:
                    correct += 1
        accuracy = correct / max(total, 1)
        return {"accuracy": accuracy}

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Entrainement
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CamemBERT NER model")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("  Entrainement CamemBERT NER natif")
    print("=" * 60)

    # --- Chargement des donnees ---
    print(f"\n[1/4] Chargement des donnees depuis {data_dir}...")
    with open(data_dir / "train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(data_dir / "val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"      Train: {len(train_data)} | Val: {len(val_data)}")

    # --- Tokenizer ---
    print("\n[2/4] Chargement du tokenizer camembert-base...")
    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

    train_dataset = NERDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = NERDataset(val_data, tokenizer, max_length=args.max_length)

    print(f"      Train apres conversion: {len(train_dataset)} | Val: {len(val_dataset)}")

    # --- Modele ---
    print("\n[3/4] Initialisation du modele CamemBERT + classification head...")
    model = CamembertForTokenClassification.from_pretrained(
        "camembert-base",
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      Device : {device}")
    print(f"      Labels : {LABEL_LIST}")
    print(f"      Params : {sum(p.numel() for p in model.parameters()):,}")

    # --- Training arguments ---
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    # --- Trainer ---
    print("\n[4/4] Lancement de l'entrainement...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # --- Sauvegarde ---
    print(f"\nSauvegarde du modele dans {MODEL_OUTPUT_DIR}...")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    # Sauvegarder la config des labels
    label_config = {
        "label_list": LABEL_LIST,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
    }
    with open(MODEL_OUTPUT_DIR / "label_config.json", "w", encoding="utf-8") as f:
        json.dump(label_config, f, indent=2)

    # --- Evaluation finale ---
    print("\nEvaluation finale sur le jeu de validation :")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n  Modele sauvegarde dans : {MODEL_OUTPUT_DIR}")
    print("  Entrainement termine.")


if __name__ == "__main__":
    main()
