"""
Language detector evaluation.
"""

import time
from typing import Any, Callable

from ..metrics import LanguageResults
from ..progress import print_progress


def evaluate_language_detectors(
    detectors: list[tuple[str, Any]],
    data: list[dict[str, Any]],
    batch_size: int = 128,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, LanguageResults]:
    """
    Evaluate all language detectors.

    Args:
        detectors: List of (name, detector) tuples
        data: List of samples with 'sentence', 'language' keys
        batch_size: Batch size for progress reporting
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary mapping detector names to LanguageResults
    """
    results: dict[str, LanguageResults] = {}
    callback = progress_callback or print_progress

    sentences = [sample["sentence"] for sample in data]
    true_languages = [sample.get("language", "UNKNOWN") for sample in data]

    for name, detector in detectors:
        print(f"  Evaluating language: {name}...")
        r = LanguageResults(name=name)

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

                is_correct = pred_lang == true_lang
                if is_correct:
                    r.correct += 1

                _update_language_stats(r, true_lang, is_correct)
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

                is_correct = pred_lang == true_lang
                if is_correct:
                    r.correct += 1

                _update_language_stats(r, true_lang, is_correct)

            callback(total, total)

        results[name] = r
        print(f"    Done. Accuracy: {r.accuracy*100:.1f}%")

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
