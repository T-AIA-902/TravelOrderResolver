"""
Evaluation module for Travel Order Resolver.

This module provides tools for evaluating NLP models:
- Intent classifiers
- Entity extractors
- Language detectors
- Combined pipelines

Usage:
    python -m src.evaluation.cli --eval-type all
"""

from .data_loader import (
    is_lowercase_sample,
    is_misspelled_sample,
    load_dataset,
    normalize_fuzzy_station,
    normalize_location,
)
from .metrics import (
    CombinedResults,
    EntityMetrics,
    EntityResults,
    IntentResults,
    LanguageResults,
    update_entity_metrics,
)
from .progress import ProgressCallback, print_progress
from .reporting import export_results_json

__all__ = [
    # Data loading
    "load_dataset",
    "normalize_location",
    "normalize_fuzzy_station",
    "is_misspelled_sample",
    "is_lowercase_sample",
    # Metrics
    "EntityMetrics",
    "IntentResults",
    "EntityResults",
    "CombinedResults",
    "LanguageResults",
    "update_entity_metrics",
    # Progress
    "ProgressCallback",
    "print_progress",
    # Reporting
    "export_results_json",
]
