"""
Training metrics for intent classification and entity extraction.

Provides compute_metrics functions compatible with HuggingFace Trainer.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_intent_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for intent classification.

    Args:
        eval_pred: EvalPrediction from Trainer (predictions, labels)

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred

    # Get predicted class
    if len(predictions.shape) > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = predictions

    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
    }


def compute_generation_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for generative entity extraction.

    For generative models, we track loss and perplexity.
    Actual entity accuracy is evaluated separately.

    Args:
        eval_pred: EvalPrediction from Trainer

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred

    # For generative models, we mainly track loss
    # More detailed metrics are computed during evaluation
    return {
        "eval_samples": len(labels) if hasattr(labels, "__len__") else 0,
    }
