"""
Evaluation du modele CamemBERT NER entraine.

Charge le modele depuis models/camembert-ner-retrain/ et evalue
sur le jeu de test (datasets/processed/ner_retrain/test.json).

Metriques :
  - Precision / Recall / F1 par entite (B-DEP, I-DEP, B-DEST, I-DEST)
  - F1 macro / micro
  - Matrice de confusion
  - Exemples de predictions

Usage:
    python models/camembert-ner-retrain/evaluate.py
    python models/camembert-ner-retrain/evaluate.py --model_dir models/camembert-ner-retrain
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from transformers import CamembertTokenizerFast, CamembertForTokenClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "camembert-ner-retrain"
DEFAULT_TEST_PATH = PROJECT_ROOT / "datasets" / "processed" / "ner_retrain" / "test.json"
RESULTS_DIR = PROJECT_ROOT / "reports"

LABEL_LIST = ["O", "B-DEP", "I-DEP", "B-DEST", "I-DEST"]
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}


def predict_sample(model, tokenizer, words, device, max_length=128):
    """Prediction NER sur une liste de mots."""
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(dim=-1).squeeze().cpu().tolist()
    if isinstance(predictions, int):
        predictions = [predictions]

    # Mapper les predictions subword -> word-level
    word_ids = encoding["input_ids"].squeeze().cpu()
    token_word_ids = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
    ).word_ids()

    word_preds = []
    prev_word_id = None
    for idx, word_id in enumerate(token_word_ids):
        if word_id is not None and word_id != prev_word_id:
            if idx < len(predictions):
                word_preds.append(ID2LABEL[predictions[idx]])
            else:
                word_preds.append("O")
        prev_word_id = word_id

    # Ajuster la longueur
    while len(word_preds) < len(words):
        word_preds.append("O")
    word_preds = word_preds[: len(words)]

    return word_preds


def compute_entity_metrics(true_labels_all, pred_labels_all):
    """Calcul des metriques par entite avec seqeval, ou manuellement."""
    try:
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

        report_str = classification_report(true_labels_all, pred_labels_all, zero_division=0)
        f1 = f1_score(true_labels_all, pred_labels_all, zero_division=0)
        precision = precision_score(true_labels_all, pred_labels_all, zero_division=0)
        recall = recall_score(true_labels_all, pred_labels_all, zero_division=0)

        return {
            "report": report_str,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    except ImportError:
        print("  [WARN] seqeval non installe, calcul token-level uniquement")
        return compute_token_metrics(true_labels_all, pred_labels_all)


def compute_token_metrics(true_labels_all, pred_labels_all):
    """Metriques token-level (fallback si seqeval absent)."""
    tp = Counter()
    fp = Counter()
    fn = Counter()
    correct = 0
    total = 0

    for true_seq, pred_seq in zip(true_labels_all, pred_labels_all):
        for t, p in zip(true_seq, pred_seq):
            total += 1
            if t == p:
                correct += 1
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1

    report_lines = [
        f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "-" * 55,
    ]
    for label in LABEL_LIST:
        p = tp[label] / max(tp[label] + fp[label], 1)
        r = tp[label] / max(tp[label] + fn[label], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        support = tp[label] + fn[label]
        report_lines.append(f"{label:<12} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>10}")

    accuracy = correct / max(total, 1)
    report_lines.append(f"\n{'Accuracy':<12} {accuracy:>10.4f} (total tokens: {total})")

    return {
        "report": "\n".join(report_lines),
        "f1": accuracy,
        "precision": accuracy,
        "recall": accuracy,
    }


def build_confusion_matrix(true_labels_all, pred_labels_all):
    """Construit une matrice de confusion token-level."""
    matrix = {t: {p: 0 for p in LABEL_LIST} for t in LABEL_LIST}
    for true_seq, pred_seq in zip(true_labels_all, pred_labels_all):
        for t, p in zip(true_seq, pred_seq):
            if t in matrix and p in matrix[t]:
                matrix[t][p] += 1
    return matrix


def format_confusion_matrix(matrix):
    """Affichage de la matrice de confusion."""
    header = f"{'':>8}" + "".join(f"{l:>8}" for l in LABEL_LIST)
    lines = [header, "-" * (8 + 8 * len(LABEL_LIST))]
    for true_label in LABEL_LIST:
        row = f"{true_label:>8}" + "".join(f"{matrix[true_label][p]:>8}" for p in LABEL_LIST)
        lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CamemBERT NER model")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--test_path", type=str, default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--show_examples", type=int, default=10, help="Number of example predictions to show")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    test_path = Path(args.test_path)

    print("=" * 60)
    print("  Evaluation CamemBERT NER natif")
    print("=" * 60)

    # Charger modele et tokenizer
    print(f"\n[1/3] Chargement du modele depuis {model_dir}...")
    tokenizer = CamembertTokenizerFast.from_pretrained(str(model_dir))
    model = CamembertForTokenClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"      Device: {device}")

    # Charger les donnees de test
    print(f"\n[2/3] Chargement du dataset de test ({test_path})...")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"      {len(test_data)} exemples")

    # Predictions
    print("\n[3/3] Predictions en cours...")
    true_labels_all = []
    pred_labels_all = []
    examples = []

    for i, sample in enumerate(test_data):
        words = sample["tokens"]
        true_labels = sample["labels"]

        pred_labels = predict_sample(model, tokenizer, words, device)

        # Aligner les longueurs
        min_len = min(len(true_labels), len(pred_labels))
        true_labels_all.append(true_labels[:min_len])
        pred_labels_all.append(pred_labels[:min_len])

        if i < args.show_examples:
            examples.append({
                "id": sample.get("id", f"sample_{i}"),
                "sentence": sample.get("sentence", " ".join(words)),
                "true": true_labels[:min_len],
                "pred": pred_labels[:min_len],
            })

    # --- Resultats ---
    print("\n" + "=" * 60)
    print("  RESULTATS")
    print("=" * 60)

    metrics = compute_entity_metrics(true_labels_all, pred_labels_all)

    print("\n--- Rapport de classification ---")
    print(metrics["report"])

    print(f"\n--- Metriques globales ---")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")

    # Matrice de confusion
    print("\n--- Matrice de confusion (token-level) ---")
    confusion = build_confusion_matrix(true_labels_all, pred_labels_all)
    print(format_confusion_matrix(confusion))

    # Exemples
    if examples:
        print(f"\n--- Exemples de predictions ({len(examples)} premiers) ---")
        for ex in examples:
            print(f"\n  [{ex['id']}] {ex['sentence']}")
            for w, t, p in zip(ex["sentence"].split(), ex["true"], ex["pred"]):
                marker = "  " if t == p else ">>"
                print(f"    {marker} {w:<25} vrai={t:<10} pred={p:<10}")

    # Sauvegarder les resultats
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model": str(model_dir),
        "test_samples": len(test_data),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": confusion,
    }
    results_path = RESULTS_DIR / "camembert_ner_retrain_eval.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultats sauvegardes dans {results_path}")


if __name__ == "__main__":
    main()
