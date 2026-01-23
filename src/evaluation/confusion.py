"""Confusion matrix utilities for evaluation.

Provides confusion matrix generation and visualization for:
- CLI (ASCII output)
- Jupyter notebooks (matplotlib heatmaps)

This module can be imported by both CLI and notebooks for consistent
confusion matrix visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure


def compute_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute confusion matrix from predictions.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional ordered list of labels. If None, inferred from data.

    Returns:
        Tuple of (confusion_matrix, labels)
        Matrix[i, j] = count of samples with true=labels[i], pred=labels[j]

    Example:
        >>> y_true = ["TRIP", "TRIP", "NOT_TRIP", "TRIP"]
        >>> y_pred = ["TRIP", "NOT_TRIP", "NOT_TRIP", "TRIP"]
        >>> matrix, labels = compute_confusion_matrix(y_true, y_pred)
        >>> print(labels)
        ['NOT_TRIP', 'TRIP']
        >>> print(matrix)
        [[1 0]
         [1 2]]
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)

    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            matrix[label_to_idx[true], label_to_idx[pred]] += 1

    return matrix, labels


def format_confusion_matrix_ascii(
    matrix: np.ndarray,
    labels: list[str],
    title: str = "",
) -> str:
    """Format confusion matrix as ASCII table.

    Args:
        matrix: Confusion matrix (n x n)
        labels: Class labels
        title: Optional title

    Returns:
        ASCII-formatted confusion matrix string

    Example output:
        Intent Classification Confusion Matrix
        ======================================
                  TRIP NOT_TRIP  UNKNOWN
           TRIP   8500      400      100
       NOT_TRIP    200     3200       50
        UNKNOWN     50       30       70
    """
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    # Determine column width
    max_label = max(len(label) for label in labels)
    col_width = max(max_label, 7)  # At least 7 chars for numbers

    # Header row (predicted labels)
    header = " " * (max_label + 2) + " ".join(f"{label:>{col_width}}" for label in labels)
    lines.append(header)

    # Data rows (true labels)
    for i, label in enumerate(labels):
        row_values = " ".join(f"{matrix[i, j]:>{col_width}}" for j in range(len(labels)))
        lines.append(f"{label:>{max_label}}  {row_values}")

    return "\n".join(lines)


def format_confusion_matrix_md(
    matrix: np.ndarray,
    labels: list[str],
    title: str = "",
) -> str:
    """Format confusion matrix as markdown table.

    Args:
        matrix: Confusion matrix (n x n)
        labels: Class labels
        title: Optional title

    Returns:
        Markdown-formatted confusion matrix string

    Example output:
        ### Intent Classification Confusion Matrix

        | True \\ Pred | TRIP | NOT_TRIP | UNKNOWN |
        |-------------|------|----------|---------|
        | TRIP        | 8500 | 400      | 100     |
        | NOT_TRIP    | 200  | 3200     | 50      |
        | UNKNOWN     | 50   | 30       | 70      |
    """
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")

    # Header row
    header = "| True \\\\ Pred | " + " | ".join(labels) + " |"
    lines.append(header)

    # Separator
    sep = "|" + "|".join(["---"] * (len(labels) + 1)) + "|"
    lines.append(sep)

    # Data rows
    for i, label in enumerate(labels):
        row_values = " | ".join(str(matrix[i, j]) for j in range(len(labels)))
        lines.append(f"| {label} | {row_values} |")

    return "\n".join(lines)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    normalize: bool = False,
) -> matplotlib.figure.Figure:
    """Plot confusion matrix as matplotlib heatmap.

    Args:
        matrix: Confusion matrix (n x n)
        labels: Class labels
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        normalize: If True, normalize by row (true class)

    Returns:
        matplotlib Figure object

    Note:
        Requires matplotlib and seaborn to be installed.
        These are optional dependencies for notebook visualization.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Work with a copy if normalizing
    display_matrix = matrix.copy()

    if normalize:
        display_matrix = display_matrix.astype(float)
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(display_matrix, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        display_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    return fig
