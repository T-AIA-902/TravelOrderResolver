"""
Reporting utilities for evaluation results.

Provides functions to print sklearn-style classification reports and export results to JSON.
"""

import json
from typing import Any

from .metrics import ClassMetrics, CombinedResults, EntityResults, IntentResults, LanguageResults


def _print_classification_report(
    title: str,
    model_name: str,
    class_metrics: list[tuple[str, ClassMetrics, int]],
    accuracy: float,
    total_support: int,
    latency_ms: float,
) -> None:
    """
    Print sklearn-style classification report.

    Args:
        title: Report title (e.g., "LANGUAGE DETECTION")
        model_name: Name of the model
        class_metrics: List of (class_name, metrics, support) tuples
        accuracy: Overall accuracy
        total_support: Total number of samples
        latency_ms: Average latency in milliseconds
    """
    print("\n" + "=" * 72)
    print(f"{title}: {model_name}")
    print("=" * 72)
    print(f"{'':14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print()

    precisions, recalls, f1s = [], [], []
    for name, m, support in class_metrics:
        print(
            f"{name:<14} {m.precision:>10.2f} {m.recall:>10.2f} "
            f"{m.f1_score:>10.2f} {support:>10}"
        )
        precisions.append(m.precision)
        recalls.append(m.recall)
        f1s.append(m.f1_score)

    print()
    print(f"{'accuracy':<14} {'':>10} {'':>10} {accuracy:>10.2f} {total_support:>10}")

    if precisions:
        macro_p = sum(precisions) / len(precisions)
        macro_r = sum(recalls) / len(recalls)
        macro_f1 = sum(f1s) / len(f1s)
        print(
            f"{'macro avg':<14} {macro_p:>10.2f} {macro_r:>10.2f} "
            f"{macro_f1:>10.2f} {total_support:>10}"
        )

    print(f"{'latency':<14} {'':>10} {'':>10} {'':>10} {latency_ms:>8.1f}ms")
    print("=" * 72)


def print_table_language(results: dict[str, LanguageResults]) -> None:
    """Print sklearn-style classification reports for language detection."""
    for name, r in results.items():
        _print_classification_report(
            title="LANGUAGE DETECTION",
            model_name=name,
            class_metrics=[
                ("fr", r.french_metrics, r.french_total),
                ("en", r.english_metrics, r.english_total),
                ("unk", r.unknown_lang_metrics, r.unknown_total),
            ],
            accuracy=r.accuracy,
            total_support=r.total,
            latency_ms=r.avg_latency_ms,
        )


def print_table_intent(results: dict[str, IntentResults]) -> None:
    """Print sklearn-style classification reports for intent classification."""
    for name, r in results.items():
        _print_classification_report(
            title="INTENT CLASSIFICATION",
            model_name=name,
            class_metrics=[
                ("TRIP", r.trip_metrics, r.trip_metrics.support),
                ("NOT_TRIP", r.not_trip_metrics, r.not_trip_metrics.support),
                ("UNKNOWN", r.unknown_intent_metrics, r.unknown_intent_metrics.support),
            ],
            accuracy=r.accuracy,
            total_support=r.total,
            latency_ms=r.avg_latency_ms,
        )


def print_table_entity(results: dict[str, EntityResults], title: str, table_num: int) -> None:
    """Print sklearn-style classification reports for entity extraction."""
    for name, r in results.items():
        _print_classification_report(
            title=title,
            model_name=name,
            class_metrics=[
                ("departure", r.departure_metrics, r.departure_metrics.support),
                ("destination", r.destination_metrics, r.destination_metrics.support),
            ],
            accuracy=r.accuracy,
            total_support=r.total,
            latency_ms=r.avg_latency_ms,
        )


def print_table_entity_combined(
    results_no_fuzzy: dict[str, EntityResults],
    results_fuzzy: dict[str, EntityResults],
) -> None:
    """Print sklearn-style classification reports for entity extraction with fuzzy comparison."""
    for name in results_no_fuzzy.keys():
        # Without fuzzy
        r = results_no_fuzzy[name]
        _print_classification_report(
            title="ENTITY EXTRACTION",
            model_name=name,
            class_metrics=[
                ("departure", r.departure_metrics, r.departure_metrics.support),
                ("destination", r.destination_metrics, r.destination_metrics.support),
            ],
            accuracy=r.accuracy,
            total_support=r.total,
            latency_ms=r.avg_latency_ms,
        )

        # With fuzzy
        fuzzy_name = f"{name} + Fuzzy"
        if fuzzy_name in results_fuzzy:
            r_fuzzy = results_fuzzy[fuzzy_name]
            _print_classification_report(
                title="ENTITY EXTRACTION",
                model_name=fuzzy_name,
                class_metrics=[
                    (
                        "departure",
                        r_fuzzy.departure_metrics,
                        r_fuzzy.departure_metrics.support,
                    ),
                    (
                        "destination",
                        r_fuzzy.destination_metrics,
                        r_fuzzy.destination_metrics.support,
                    ),
                ],
                accuracy=r_fuzzy.accuracy,
                total_support=r_fuzzy.total,
                latency_ms=r_fuzzy.avg_latency_ms,
            )


def print_table_combined(results: list[CombinedResults], title: str, table_num: int) -> None:
    """Print combined evaluation table."""
    print("\n" + "=" * 72)
    print(f"{title}")
    print("=" * 72)
    print(
        f"{'Intent':<12} {'Entity':<12} {'Fuzzy':<6} "
        f"{'Intent Acc':>10} {'Entity Acc':>10} {'Latency':>10}"
    )
    print()

    for r in results:
        fuzzy_marker = "Yes" if r.with_fuzzy else "-"
        print(
            f"{r.intent_name:<12} {r.entity_name:<12} {fuzzy_marker:<6} "
            f"{r.intent_accuracy:>10.2f} {r.entity_accuracy:>10.2f} "
            f"{r.avg_latency_ms:>8.1f}ms"
        )

    print("=" * 72)


def export_results_json(
    intent_results: dict[str, IntentResults],
    entity_results: dict[str, EntityResults],
    entity_fuzzy_results: dict[str, EntityResults],
    combined_results: list[CombinedResults],
    combined_fuzzy_results: list[CombinedResults],
    language_results: dict[str, LanguageResults],
    output_path: str,
) -> None:
    """Export all results to JSON file."""
    export_data: dict[str, Any] = {
        "intent": {
            name: {
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "trip_precision": r.trip_metrics.precision,
                "trip_recall": r.trip_metrics.recall,
                "trip_f1": r.trip_metrics.f1_score,
                "trip_support": r.trip_metrics.support,
                "not_trip_precision": r.not_trip_metrics.precision,
                "not_trip_recall": r.not_trip_metrics.recall,
                "not_trip_f1": r.not_trip_metrics.f1_score,
                "not_trip_support": r.not_trip_metrics.support,
                "unknown_precision": r.unknown_intent_metrics.precision,
                "unknown_recall": r.unknown_intent_metrics.recall,
                "unknown_f1": r.unknown_intent_metrics.f1_score,
                "unknown_support": r.unknown_intent_metrics.support,
                "latency_ms": r.avg_latency_ms,
            }
            for name, r in intent_results.items()
        },
        "entity": {
            name: {
                "accuracy": r.accuracy,
                "macro_f1": r.avg_f1,
                "departure_precision": r.departure_metrics.precision,
                "departure_recall": r.departure_metrics.recall,
                "departure_f1": r.departure_metrics.f1_score,
                "departure_support": r.departure_metrics.support,
                "destination_precision": r.destination_metrics.precision,
                "destination_recall": r.destination_metrics.recall,
                "destination_f1": r.destination_metrics.f1_score,
                "destination_support": r.destination_metrics.support,
                "latency_ms": r.avg_latency_ms,
            }
            for name, r in entity_results.items()
        },
        "entity_fuzzy": {
            name: {
                "accuracy": r.accuracy,
                "macro_f1": r.avg_f1,
                "departure_precision": r.departure_metrics.precision,
                "departure_recall": r.departure_metrics.recall,
                "departure_f1": r.departure_metrics.f1_score,
                "departure_support": r.departure_metrics.support,
                "destination_precision": r.destination_metrics.precision,
                "destination_recall": r.destination_metrics.recall,
                "destination_f1": r.destination_metrics.f1_score,
                "destination_support": r.destination_metrics.support,
                "latency_ms": r.avg_latency_ms,
            }
            for name, r in entity_fuzzy_results.items()
        },
        "combined": [
            {
                "intent": r.intent_name,
                "entity": r.entity_name,
                "with_fuzzy": r.with_fuzzy,
                "intent_accuracy": r.intent_accuracy,
                "entity_accuracy": r.entity_accuracy,
                "latency_ms": r.avg_latency_ms,
            }
            for r in combined_results
        ],
        "combined_fuzzy": [
            {
                "intent": r.intent_name,
                "entity": r.entity_name,
                "with_fuzzy": r.with_fuzzy,
                "intent_accuracy": r.intent_accuracy,
                "entity_accuracy": r.entity_accuracy,
                "latency_ms": r.avg_latency_ms,
            }
            for r in combined_fuzzy_results
        ],
        "language": {
            name: {
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "french_precision": r.french_metrics.precision,
                "french_recall": r.french_metrics.recall,
                "french_f1": r.french_metrics.f1_score,
                "french_support": r.french_total,
                "english_precision": r.english_metrics.precision,
                "english_recall": r.english_metrics.recall,
                "english_f1": r.english_metrics.f1_score,
                "english_support": r.english_total,
                "unknown_precision": r.unknown_lang_metrics.precision,
                "unknown_recall": r.unknown_lang_metrics.recall,
                "unknown_f1": r.unknown_lang_metrics.f1_score,
                "unknown_support": r.unknown_total,
                "latency_ms": r.avg_latency_ms,
            }
            for name, r in language_results.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
