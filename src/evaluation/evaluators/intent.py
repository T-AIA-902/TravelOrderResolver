"""
Intent classifier evaluation.
"""

import time
from typing import Any, Callable

from ..metrics import IntentResults
from ..progress import print_progress


def evaluate_intent_classifiers(
    classifiers: list[tuple[str, Any]],
    data: list[dict[str, Any]],
    batch_size: int = 128,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, IntentResults]:
    """
    Evaluate all intent classifiers.

    Args:
        classifiers: List of (name, classifier) tuples
        data: List of samples with 'sentence', 'intent', 'language' keys
        batch_size: Batch size for batched inference
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary mapping classifier names to IntentResults
    """
    results: dict[str, IntentResults] = {}
    callback = progress_callback or print_progress

    sentences = [sample["sentence"] for sample in data]
    true_intents = [sample["intent"] for sample in data]
    languages = [sample.get("language", "UNKNOWN") for sample in data]

    for name, classifier in classifiers:
        print(f"  Evaluating intent: {name}...")
        r = IntentResults(name=name)

        if hasattr(classifier, "classify_batch"):
            print(f"    Using batched inference (batch_size={batch_size})...")
            start = time.perf_counter()
            predictions = classifier.classify_batch(
                sentences,
                batch_size=batch_size,
                progress_callback=callback,
            )
            end = time.perf_counter()

            total_time_ms = (end - start) * 1000
            avg_latency = total_time_ms / len(sentences) if sentences else 0

            for pred, true_intent, lang in zip(predictions, true_intents, languages):
                pred_intent, _ = pred
                r.latencies.append(avg_latency)
                r.total += 1

                is_correct = pred_intent == true_intent
                if is_correct:
                    r.correct += 1

                _update_language_stats(r, lang, is_correct)
                _update_class_metrics(r, pred_intent, true_intent)
        else:
            # Sequential classification
            total = len(sentences)
            for i, (sentence, true_intent, lang) in enumerate(
                zip(sentences, true_intents, languages)
            ):
                if i % batch_size == 0:
                    callback(i, total)

                start = time.perf_counter()
                pred_intent, _ = classifier.classify(sentence)
                end = time.perf_counter()

                r.latencies.append((end - start) * 1000)
                r.total += 1

                is_correct = pred_intent == true_intent
                if is_correct:
                    r.correct += 1

                _update_language_stats(r, lang, is_correct)
                _update_class_metrics(r, pred_intent, true_intent)

            callback(total, total)

        results[name] = r
        print(f"    Done. Accuracy: {r.accuracy*100:.1f}%")

    return results


def _update_language_stats(r: IntentResults, lang: str, is_correct: bool) -> None:
    """Update per-language statistics."""
    if lang == "FRENCH":
        r.french_total += 1
        if is_correct:
            r.french_correct += 1
    elif lang == "ENGLISH":
        r.english_total += 1
        if is_correct:
            r.english_correct += 1
    else:
        r.unknown_total += 1
        if is_correct:
            r.unknown_correct += 1


def _update_class_metrics(r: IntentResults, pred_intent: str, true_intent: str) -> None:
    """Update per-class TP/FP/FN metrics for intent classification."""
    classes = {
        "TRIP": r.trip_metrics,
        "NOT_TRIP": r.not_trip_metrics,
        "UNKNOWN": r.unknown_intent_metrics,
    }

    for cls, m in classes.items():
        if pred_intent == cls and true_intent == cls:
            m.tp += 1
        elif pred_intent == cls and true_intent != cls:
            m.fp += 1
        elif pred_intent != cls and true_intent == cls:
            m.fn += 1
        else:
            m.tn += 1
