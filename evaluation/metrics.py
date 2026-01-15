"""
Metrics calculation for NLP evaluation.

This module provides functions to calculate standard NLP metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Latency
"""

import time
from typing import Dict, List, Tuple


def calculate_metrics(
    true_positives: int, false_positives: int, false_negatives: int, true_negatives: int = 0
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1-score, and accuracy.

    Args:
        true_positives: Number of correct positive predictions
        false_positives: Number of incorrect positive predictions
        false_negatives: Number of missed positive cases
        true_negatives: Number of correct negative predictions (optional)

    Returns:
        Dictionary with precision, recall, f1_score, and accuracy
    """
    total = true_positives + false_positives + false_negatives + true_negatives

    # Precision: Of all predicted positives, how many are correct?
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    # Recall: Of all actual positives, how many did we find?
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    # F1-Score: Harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Accuracy: Overall correctness
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0

    return {
        "precision": precision * 100,  # Convert to percentage
        "recall": recall * 100,
        "f1_score": f1_score * 100,
        "accuracy": accuracy * 100,
    }


def calculate_entity_metrics(
    predictions: List[Tuple[str, str]], ground_truth: List[Tuple[str, str]]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for entity extraction (departure and destination).

    Args:
        predictions: List of (departure, destination) predictions
        ground_truth: List of (departure, destination) ground truth

    Returns:
        Dictionary with metrics for departure, destination, and both
    """
    assert len(predictions) == len(
        ground_truth
    ), "Predictions and ground truth must have same length"

    total = len(predictions)

    # Counters for departure
    dep_tp = 0  # True positives
    dep_fp = 0  # False positives
    dep_fn = 0  # False negatives

    # Counters for destination
    dest_tp = 0
    dest_fp = 0
    dest_fn = 0

    # Counters for both correct
    both_correct = 0

    for (pred_dep, pred_dest), (true_dep, true_dest) in zip(predictions, ground_truth):
        # Departure metrics
        if pred_dep is not None and true_dep is not None:
            if pred_dep == true_dep:
                dep_tp += 1
            else:
                dep_fp += 1
                dep_fn += 1
        elif pred_dep is not None and true_dep is None:
            dep_fp += 1
        elif pred_dep is None and true_dep is not None:
            dep_fn += 1

        # Destination metrics
        if pred_dest is not None and true_dest is not None:
            if pred_dest == true_dest:
                dest_tp += 1
            else:
                dest_fp += 1
                dest_fn += 1
        elif pred_dest is not None and true_dest is None:
            dest_fp += 1
        elif pred_dest is None and true_dest is not None:
            dest_fn += 1

        # Both correct
        if pred_dep == true_dep and pred_dest == true_dest:
            both_correct += 1

    # Calculate metrics for departure
    dep_metrics = calculate_metrics(dep_tp, dep_fp, dep_fn)

    # Calculate metrics for destination
    dest_metrics = calculate_metrics(dest_tp, dest_fp, dest_fn)

    # Calculate metrics for both
    # For "both correct", we treat it as binary classification
    both_tp = both_correct
    both_fp = 0  # We don't have false positives in this context
    both_fn = total - both_correct
    both_metrics = calculate_metrics(both_tp, both_fp, both_fn)

    return {"departure": dep_metrics, "destination": dest_metrics, "both": both_metrics}


def measure_latency(func, *args, n_runs: int = 10, **kwargs) -> Dict[str, float]:
    """
    Measure the latency of a function over multiple runs.

    Args:
        func: Function to measure
        *args: Positional arguments for the function
        n_runs: Number of runs to average over
        **kwargs: Keyword arguments for the function

    Returns:
        Dictionary with avg_ms, min_ms, max_ms latency in milliseconds
    """
    latencies = []

    # Warm-up run (not counted)
    func(*args, **kwargs)

    # Measure runs
    for _ in range(n_runs):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return {
        "avg_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "std_ms": (
            sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)
        )
        ** 0.5,
    }


def format_metrics_table(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Format metrics as a markdown table.

    Args:
        metrics: Dictionary with metrics for different entities

    Returns:
        Formatted markdown table string
    """
    dep = metrics["departure"]
    dest = metrics["destination"]
    both = metrics["both"]
    table = "| Metric | Departure | Destination | Both Correct |\n"
    table += "|--------|-----------|-------------|-------------|\n"
    table += f"| **Precision** | {dep['precision']:.2f}% | {dest['precision']:.2f}% | {both['precision']:.2f}% |\n"  # noqa: E501
    table += f"| **Recall** | {dep['recall']:.2f}% | {dest['recall']:.2f}% | {both['recall']:.2f}% |\n"  # noqa: E501
    table += f"| **F1-Score** | {dep['f1_score']:.2f}% | {dest['f1_score']:.2f}% | {both['f1_score']:.2f}% |\n"  # noqa: E501
    table += f"| **Accuracy** | {dep['accuracy']:.2f}% | {dest['accuracy']:.2f}% | {both['accuracy']:.2f}% |\n"  # noqa: E501

    return table
