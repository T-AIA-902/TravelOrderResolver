"""Formatting utilities for evaluation output.

This module provides formatting functions that can be imported by:
- CLI (src/evaluation/cli.py)
- Notebooks (notebooks/evaluation.ipynb)

Both CLI and notebook use the same core formatting logic (DRY principle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .metrics import ClassMetrics, EntityResults, IntentResults, LanguageResults


def format_table_md(headers: list[str], rows: list[list[Any]], title: str = "") -> str:
    """Format data as a markdown table.

    Args:
        headers: Column headers
        rows: List of rows, each row is a list of cell values
        title: Optional title (rendered as ### heading)

    Returns:
        Markdown-formatted table string

    Example:
        >>> headers = ["Model", "Accuracy", "Latency"]
        >>> rows = [["Regex", "65.8%", "0.1ms"], ["SpaCy", "79.1%", "1.0ms"]]
        >>> print(format_table_md(headers, rows, "Intent Classification"))
        ### Intent Classification

        | Model | Accuracy | Latency |
        |---|---|---|
        | Regex | 65.8% | 0.1ms |
        | SpaCy | 79.1% | 1.0ms |
    """
    lines = []

    if title:
        lines.append(f"### {title}")
        lines.append("")

    # Header row
    lines.append("| " + " | ".join(headers) + " |")

    # Separator row
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    # Data rows
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return "\n".join(lines)


def format_accuracy(value: float, decimals: int = 1) -> str:
    """Format accuracy as percentage string.

    Args:
        value: Accuracy value (0.0 to 1.0)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string (e.g., "79.1%")
    """
    return f"{value:.{decimals}%}"


def format_latency(value_ms: float, decimals: int = 1) -> str:
    """Format latency as milliseconds string.

    Args:
        value_ms: Latency in milliseconds
        decimals: Number of decimal places

    Returns:
        Formatted latency string (e.g., "5.4ms")
    """
    return f"{value_ms:.{decimals}f}ms"


def format_f1(value: float, decimals: int = 2) -> str:
    """Format F1 score.

    Args:
        value: F1 score (0.0 to 1.0)
        decimals: Number of decimal places

    Returns:
        Formatted F1 string (e.g., "0.85")
    """
    return f"{value:.{decimals}f}"


# =============================================================================
# Sklearn-style classification report formatting
# =============================================================================


def _compute_macro_averages(
    class_metrics: list[tuple[str, ClassMetrics, int]],
) -> tuple[float, float, float]:
    """Compute macro-averaged precision, recall, F1 from class metrics.

    Args:
        class_metrics: List of (class_name, metrics, support) tuples

    Returns:
        Tuple of (macro_precision, macro_recall, macro_f1)
    """
    if not class_metrics:
        return 0.0, 0.0, 0.0

    precisions = [m.precision for _, m, _ in class_metrics]
    recalls = [m.recall for _, m, _ in class_metrics]
    f1s = [m.f1_score for _, m, _ in class_metrics]

    return (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1s) / len(f1s),
    )


def format_classification_report_md(
    title: str,
    class_metrics: list[tuple[str, ClassMetrics, int]],
    accuracy: float,
    total_support: int,
    latency_ms: float,
) -> str:
    """Format sklearn-style classification report as markdown.

    Args:
        title: Report title (e.g., "Language Detection: Langdetect")
        class_metrics: List of (class_name, metrics, support) tuples
        accuracy: Overall accuracy
        total_support: Total number of samples
        latency_ms: Average latency in milliseconds

    Returns:
        Markdown-formatted classification report

    Example output:
        ### Language Detection: Langdetect

        |              | Precision | Recall |   F1 | Support |
        |--------------|-----------|--------|------|---------|
        | fr           |      0.96 |   0.80 | 0.87 |   13366 |
        | en           |      0.48 |   0.66 | 0.55 |     966 |
        | unk          |      0.14 |   0.55 | 0.23 |     673 |
        |              |           |        |      |         |
        | accuracy     |           |        | 0.78 |   15005 |
        | macro avg    |      0.53 |   0.67 | 0.55 |   15005 |
        | latency      |           |        |      |   6.0ms |
    """
    lines = [f"### {title}", ""]
    lines.append("|              | Precision | Recall |   F1 | Support |")
    lines.append("|--------------|-----------|--------|------|---------|")

    for name, m, support in class_metrics:
        row = (
            f"| {name:<12} | {m.precision:>9.2f} | {m.recall:>6.2f} "
            f"| {m.f1_score:>4.2f} | {support:>7} |"
        )
        lines.append(row)

    lines.append("|              |           |        |      |         |")
    acc_row = (
        f"| {'accuracy':<12} | {'':>9} | {'':>6} " f"| {accuracy:>4.2f} | {total_support:>7} |"
    )
    lines.append(acc_row)

    macro_p, macro_r, macro_f1 = _compute_macro_averages(class_metrics)
    macro_row = (
        f"| {'macro avg':<12} | {macro_p:>9.2f} | {macro_r:>6.2f} "
        f"| {macro_f1:>4.2f} | {total_support:>7} |"
    )
    lines.append(macro_row)
    lines.append(f"| {'latency':<12} | {'':>9} | {'':>6} | {'':>4} | {latency_ms:>5.1f}ms |")

    return "\n".join(lines)


def format_classification_report_ascii(
    title: str,
    class_metrics: list[tuple[str, ClassMetrics, int]],
    accuracy: float,
    total_support: int,
    latency_ms: float,
) -> str:
    """Format sklearn-style classification report as ASCII (for CLI).

    Same structure as format_classification_report_md but with ASCII borders.

    Args:
        title: Report title (e.g., "LANGUAGE DETECTION: Langdetect")
        class_metrics: List of (class_name, metrics, support) tuples
        accuracy: Overall accuracy
        total_support: Total number of samples
        latency_ms: Average latency in milliseconds

    Returns:
        ASCII-formatted classification report
    """
    lines = []
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)
    lines.append(f"{'':14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("")

    for name, m, support in class_metrics:
        lines.append(
            f"{name:<14} {m.precision:>10.2f} {m.recall:>10.2f} {m.f1_score:>10.2f} {support:>10}"
        )

    lines.append("")
    lines.append(f"{'accuracy':<14} {'':>10} {'':>10} {accuracy:>10.2f} {total_support:>10}")

    macro_p, macro_r, macro_f1 = _compute_macro_averages(class_metrics)
    macro_row = (
        f"{'macro avg':<14} {macro_p:>10.2f} {macro_r:>10.2f} "
        f"{macro_f1:>10.2f} {total_support:>10}"
    )
    lines.append(macro_row)
    lines.append(f"{'latency':<14} {'':>10} {'':>10} {'':>10} {latency_ms:>8.1f}ms")
    lines.append("=" * 72)

    return "\n".join(lines)


# =============================================================================
# Convenience wrappers for specific result types
# =============================================================================


def format_language_report_md(name: str, result: LanguageResults) -> str:
    """Format LanguageResults as sklearn-style markdown report."""
    return format_classification_report_md(
        title=f"Language Detection: {name}",
        class_metrics=[
            ("fr", result.french_metrics, result.french_total),
            ("en", result.english_metrics, result.english_total),
            ("unk", result.unknown_lang_metrics, result.unknown_total),
        ],
        accuracy=result.accuracy,
        total_support=result.total,
        latency_ms=result.avg_latency_ms,
    )


def format_intent_report_md(name: str, result: IntentResults) -> str:
    """Format IntentResults as sklearn-style markdown report."""
    return format_classification_report_md(
        title=f"Intent Classification: {name}",
        class_metrics=[
            ("TRIP", result.trip_metrics, result.trip_metrics.support),
            ("NOT_TRIP", result.not_trip_metrics, result.not_trip_metrics.support),
            ("UNKNOWN", result.unknown_intent_metrics, result.unknown_intent_metrics.support),
        ],
        accuracy=result.accuracy,
        total_support=result.total,
        latency_ms=result.avg_latency_ms,
    )


def format_entity_report_md(name: str, result: EntityResults) -> str:
    """Format EntityResults as sklearn-style markdown report."""
    return format_classification_report_md(
        title=f"Entity Extraction: {name}",
        class_metrics=[
            ("departure", result.departure_metrics, result.departure_metrics.support),
            ("destination", result.destination_metrics, result.destination_metrics.support),
        ],
        accuracy=result.accuracy,
        total_support=result.total,
        latency_ms=result.avg_latency_ms,
    )


def format_language_report_ascii(name: str, result: LanguageResults) -> str:
    """Format LanguageResults as sklearn-style ASCII report (for CLI)."""
    return format_classification_report_ascii(
        title=f"LANGUAGE DETECTION: {name}",
        class_metrics=[
            ("fr", result.french_metrics, result.french_total),
            ("en", result.english_metrics, result.english_total),
            ("unk", result.unknown_lang_metrics, result.unknown_total),
        ],
        accuracy=result.accuracy,
        total_support=result.total,
        latency_ms=result.avg_latency_ms,
    )


def format_intent_report_ascii(name: str, result: IntentResults) -> str:
    """Format IntentResults as sklearn-style ASCII report (for CLI)."""
    return format_classification_report_ascii(
        title=f"INTENT CLASSIFICATION: {name}",
        class_metrics=[
            ("TRIP", result.trip_metrics, result.trip_metrics.support),
            ("NOT_TRIP", result.not_trip_metrics, result.not_trip_metrics.support),
            ("UNKNOWN", result.unknown_intent_metrics, result.unknown_intent_metrics.support),
        ],
        accuracy=result.accuracy,
        total_support=result.total,
        latency_ms=result.avg_latency_ms,
    )


def format_entity_report_ascii(name: str, result: EntityResults) -> str:
    """Format EntityResults as sklearn-style ASCII report (for CLI)."""
    return format_classification_report_ascii(
        title=f"ENTITY EXTRACTION: {name}",
        class_metrics=[
            ("departure", result.departure_metrics, result.departure_metrics.support),
            ("destination", result.destination_metrics, result.destination_metrics.support),
        ],
        accuracy=result.accuracy,
        total_support=result.total,
        latency_ms=result.avg_latency_ms,
    )
