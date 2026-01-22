"""
Reporting utilities for evaluation results.

Provides functions to print tables and export results to JSON.
"""

import json
from typing import Any

from .metrics import CombinedResults, EntityResults, IntentResults, LanguageResults


def print_table_language(results: dict[str, LanguageResults]) -> None:
    """Print Table 1: Language Detection."""
    print("\n" + "=" * 80)
    print("TABLE 1: LANGUAGE DETECTION")
    print("=" * 80)
    print("\n| Model | Overall | FR | EN | UNK | Latency |")
    print("|-------|---------|----|----|-----|---------|")

    for name, r in results.items():
        print(
            f"| {name:<10} | {r.accuracy*100:>6.1f}% | "
            f"{r.french_accuracy*100:>3.0f}% | {r.english_accuracy*100:>3.0f}% | "
            f"{r.unknown_accuracy*100:>3.0f}% | {r.avg_latency_ms:>6.2f}ms |"
        )


def print_table_intent(results: dict[str, IntentResults]) -> None:
    """Print Table 2: Intent Classification (per-language)."""
    print("\n" + "=" * 80)
    print("TABLE 2: INTENT CLASSIFICATION (per-language)")
    print("=" * 80)
    print("\n| Model | Overall | FR | EN | UNK | Latency |")
    print("|-------|---------|----|----|-----|---------|")

    for name, r in results.items():
        print(
            f"| {name:<10} | {r.accuracy*100:>6.1f}% | "
            f"{r.french_accuracy*100:>3.0f}% | {r.english_accuracy*100:>3.0f}% | "
            f"{r.unknown_accuracy*100:>3.0f}% | {r.avg_latency_ms:>6.2f}ms |"
        )


def print_table_entity(results: dict[str, EntityResults], title: str, table_num: int) -> None:
    """Print entity extraction table (legacy format)."""
    print("\n" + "=" * 80)
    print(f"TABLE {table_num}: {title}")
    print("=" * 80)
    print("\n| Model | Accuracy | Precision | Recall | F1-Score | Latency |")
    print("|-------|----------|-----------|--------|----------|---------|")

    for name, r in results.items():
        print(
            f"| {name:<20} | {r.accuracy*100:>7.1f}% | "
            f"{r.avg_precision*100:>8.1f}% | {r.avg_recall*100:>5.1f}% | "
            f"{r.avg_f1*100:>7.1f}% | {r.avg_latency_ms:>6.1f}ms |"
        )


def print_table_entity_combined(
    results_no_fuzzy: dict[str, EntityResults],
    results_fuzzy: dict[str, EntityResults],
) -> None:
    """Print Table 3: Entity Extraction with Fuzzy column."""
    print("\n" + "=" * 80)
    print("TABLE 3: ENTITY EXTRACTION")
    print("=" * 80)
    print("\n| Model | Fuzzy | Accuracy | Precision | Recall | F1 | Latency |")
    print("|-------|-------|----------|-----------|--------|-----|---------|")

    model_names = list(results_no_fuzzy.keys())

    for name in model_names:
        # Without fuzzy
        r = results_no_fuzzy[name]
        print(
            f"| {name:<10} | -     | {r.accuracy*100:>7.1f}% | "
            f"{r.avg_precision*100:>8.1f}% | {r.avg_recall*100:>5.1f}% | "
            f"{r.avg_f1*100:>3.0f}% | {r.avg_latency_ms:>6.1f}ms |"
        )

        # With fuzzy
        fuzzy_name = f"{name} + Fuzzy"
        if fuzzy_name in results_fuzzy:
            r_fuzzy = results_fuzzy[fuzzy_name]
            print(
                f"| {name:<10} | ✓     | {r_fuzzy.accuracy*100:>7.1f}% | "
                f"{r_fuzzy.avg_precision*100:>8.1f}% | {r_fuzzy.avg_recall*100:>5.1f}% | "
                f"{r_fuzzy.avg_f1*100:>3.0f}% | {r_fuzzy.avg_latency_ms:>6.1f}ms |"
            )


def print_table_combined(results: list[CombinedResults], title: str, table_num: int) -> None:
    """Print combined evaluation table."""
    print("\n" + "=" * 80)
    print(f"TABLE {table_num}: {title}")
    print("=" * 80)
    print("\n| Intent | Entity | Fuzzy | Intent Acc | Entity Acc | Latency |")
    print("|--------|--------|-------|------------|------------|---------|")

    for r in results:
        fuzzy_marker = "✓" if r.with_fuzzy else "-"
        print(
            f"| {r.intent_name:<10} | {r.entity_name:<10} | {fuzzy_marker:<5} | "
            f"{r.intent_accuracy*100:>9.1f}% | {r.entity_accuracy*100:>9.1f}% | "
            f"{r.avg_latency_ms:>6.1f}ms |"
        )


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
            name: {"accuracy": r.accuracy, "latency_ms": r.avg_latency_ms}
            for name, r in intent_results.items()
        },
        "entity": {
            name: {
                "accuracy": r.accuracy,
                "precision": r.avg_precision,
                "recall": r.avg_recall,
                "f1": r.avg_f1,
                "latency_ms": r.avg_latency_ms,
            }
            for name, r in entity_results.items()
        },
        "entity_fuzzy": {
            name: {
                "accuracy": r.accuracy,
                "precision": r.avg_precision,
                "recall": r.avg_recall,
                "f1": r.avg_f1,
                "latency_ms": r.avg_latency_ms,
            }
            for name, r in entity_fuzzy_results.items()
        },
        "combined": [
            {
                "intent": r.intent_name,
                "entity": r.entity_name,
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
                "intent_accuracy": r.intent_accuracy,
                "entity_accuracy": r.entity_accuracy,
                "latency_ms": r.avg_latency_ms,
            }
            for r in combined_fuzzy_results
        ],
        "language": {
            name: {
                "accuracy": r.accuracy,
                "french_accuracy": r.french_accuracy,
                "english_accuracy": r.english_accuracy,
                "unknown_accuracy": r.unknown_accuracy,
                "latency_ms": r.avg_latency_ms,
            }
            for name, r in language_results.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
