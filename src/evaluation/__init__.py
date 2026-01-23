"""
Evaluation module for Travel Order Resolver.

This module provides tools for evaluating NLP models:
- Intent classifiers
- Entity extractors
- Language detectors
- Combined pipelines
- Confusion matrices

Usage:
    python -m src.evaluation.cli --eval-type all

    # Or import reusable functions:
    from src.evaluation import (
        load_dataset,
        apply_preprocessing,
        create_language_detectors,
        create_intent_classifiers,
        create_entity_extractors,
        format_table_md,
        format_intent_report_md,
        compute_confusion_matrix,
        plot_confusion_matrix,
    )
"""

from .confusion import (
    compute_confusion_matrix,
    format_confusion_matrix_ascii,
    format_confusion_matrix_md,
    plot_confusion_matrix,
)
from .data_loader import (
    is_lowercase_sample,
    is_misspelled_sample,
    load_dataset,
    normalize_fuzzy_station,
    normalize_location,
)
from .formatting import (
    format_accuracy,
    format_classification_report_ascii,
    format_classification_report_md,
    format_entity_report_ascii,
    format_entity_report_md,
    format_f1,
    format_intent_report_ascii,
    format_intent_report_md,
    format_language_report_ascii,
    format_language_report_md,
    format_latency,
    format_table_md,
)
from .metrics import (
    CombinedResults,
    EntityMetrics,
    EntityResults,
    IntentResults,
    LanguageResults,
    update_entity_metrics,
)
from .model_factory import (
    DeviceType,
    create_entity_extractors,
    create_fuzzy_post_processor,
    create_intent_classifiers,
    create_language_detectors,
)
from .preprocessing import apply_preprocessing
from .progress import ProgressCallback, print_progress
from .reporting import export_results_json

__all__ = [
    # Data loading
    "load_dataset",
    "normalize_location",
    "normalize_fuzzy_station",
    "is_misspelled_sample",
    "is_lowercase_sample",
    # Preprocessing
    "apply_preprocessing",
    # Model factory
    "DeviceType",
    "create_language_detectors",
    "create_intent_classifiers",
    "create_entity_extractors",
    "create_fuzzy_post_processor",
    # Formatting - basic
    "format_table_md",
    "format_accuracy",
    "format_latency",
    "format_f1",
    # Formatting - sklearn-style reports (shared by CLI and notebooks)
    "format_classification_report_md",
    "format_classification_report_ascii",
    "format_language_report_md",
    "format_language_report_ascii",
    "format_intent_report_md",
    "format_intent_report_ascii",
    "format_entity_report_md",
    "format_entity_report_ascii",
    # Confusion matrix
    "compute_confusion_matrix",
    "format_confusion_matrix_ascii",
    "format_confusion_matrix_md",
    "plot_confusion_matrix",
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
