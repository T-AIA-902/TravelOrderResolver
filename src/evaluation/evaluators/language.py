"""Language detector evaluation."""

from __future__ import annotations

import time
from typing import Any, Callable, Literal, overload

from ..metrics import LanguageResults
from ..progress import print_progress

# Type alias for predictions dict
PredictionsDict = dict[str, tuple[list[str], list[str]]]


@overload
def evaluate_language_detectors(  # type: ignore[overload-overlap]
    detectors: list[tuple[str, Any]],
    data: list[dict[str, Any]],
    batch_size: int = ...,
    progress_callback: Callable[[int, int], None] | None = ...,
    return_predictions: Literal[False] = ...,
) -> dict[str, LanguageResults]:
    """Evaluate language detectors (return_predictions=False)."""
    ...


@overload
def evaluate_language_detectors(
    detectors: list[tuple[str, Any]],
    data: list[dict[str, Any]],
    batch_size: int = ...,
    progress_callback: Callable[[int, int], None] | None = ...,
    return_predictions: Literal[True] = ...,
) -> tuple[dict[str, LanguageResults], PredictionsDict]:
    """Evaluate language detectors (return_predictions=True)."""
    ...


def evaluate_language_detectors(
    detectors: list[tuple[str, Any]],
    data: list[dict[str, Any]],
    batch_size: int = 128,
    progress_callback: Callable[[int, int], None] | None = None,
    return_predictions: bool = False,
) -> dict[str, LanguageResults] | tuple[dict[str, LanguageResults], PredictionsDict]:
    """
    Evaluate all language detectors.

    Args:
        detectors: List of (name, detector) tuples
        data: List of samples with 'sentence', 'language' keys
        batch_size: Batch size for progress reporting
        progress_callback: Optional callback for progress updates
        return_predictions: If True, also return y_true/y_pred for confusion matrices

    Returns:
        If return_predictions=False: Dictionary mapping detector names to LanguageResults
        If return_predictions=True: Tuple of (results_dict, predictions_dict)
            where predictions_dict[name] = (y_true, y_pred)
    """
    results: dict[str, LanguageResults] = {}
    all_predictions: PredictionsDict = {}
    callback = progress_callback or print_progress

    sentences = [sample["sentence"] for sample in data]
    true_languages = [sample.get("language", "UNKNOWN") for sample in data]

    for name, detector in detectors:
        print(f"  Evaluating language: {name}...")
        r = LanguageResults(name=name)
        y_true: list[str] = []
        y_pred: list[str] = []

        if hasattr(detector, "detect_batch"):
            start = time.perf_counter()
            predictions = detector.detect_batch(sentences)
            end = time.perf_counter()

            total_time_ms = (end - start) * 1000
            avg_latency = total_time_ms / len(sentences) if sentences else 0

            for pred, true_lang in zip(predictions, true_languages):
                pred_lang, _ = pred
                r.latencies.append(avg_latency)
                r.total += 1

                # Store predictions for confusion matrix
                y_true.append(true_lang)
                y_pred.append(pred_lang)

                is_correct = pred_lang == true_lang
                if is_correct:
                    r.correct += 1

                _update_language_stats(r, true_lang, is_correct)
                _update_lang_class_metrics(r, pred_lang, true_lang)
        else:
            # Sequential detection
            total = len(sentences)
            for i, (sentence, true_lang) in enumerate(zip(sentences, true_languages)):
                if i % batch_size == 0:
                    callback(i, total)

                start = time.perf_counter()
                pred_lang, _ = detector.detect(sentence)
                end = time.perf_counter()

                r.latencies.append((end - start) * 1000)
                r.total += 1

                # Store predictions for confusion matrix
                y_true.append(true_lang)
                y_pred.append(pred_lang)

                is_correct = pred_lang == true_lang
                if is_correct:
                    r.correct += 1

                _update_language_stats(r, true_lang, is_correct)
                _update_lang_class_metrics(r, pred_lang, true_lang)

            callback(total, total)

        results[name] = r
        all_predictions[name] = (y_true, y_pred)
        print(f"    Done. Accuracy: {r.accuracy*100:.1f}%")

    if return_predictions:
        return results, all_predictions
    return results


def _update_language_stats(r: LanguageResults, true_lang: str, is_correct: bool) -> None:
    """Update per-language statistics."""
    if true_lang == "FRENCH":
        r.french_total += 1
        if is_correct:
            r.french_correct += 1
    elif true_lang == "ENGLISH":
        r.english_total += 1
        if is_correct:
            r.english_correct += 1
    else:
        r.unknown_total += 1
        if is_correct:
            r.unknown_correct += 1


def _update_lang_class_metrics(r: LanguageResults, pred_lang: str, true_lang: str) -> None:
    """Update per-class TP/FP/FN metrics for language detection."""
    classes = {
        "FRENCH": r.french_metrics,
        "ENGLISH": r.english_metrics,
        "UNKNOWN": r.unknown_lang_metrics,
    }

    for cls, m in classes.items():
        if pred_lang == cls and true_lang == cls:
            m.tp += 1
        elif pred_lang == cls and true_lang != cls:
            m.fp += 1
        elif pred_lang != cls and true_lang == cls:
            m.fn += 1
        else:
            m.tn += 1
