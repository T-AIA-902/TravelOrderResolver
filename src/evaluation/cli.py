"""
Command-line interface for evaluation.

Usage:
    python -m src.evaluation.cli --eval-type all
    python -m src.evaluation.cli --eval-type intent --models regex
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from .data_loader import load_dataset
from .evaluators import (
    evaluate_combined,
    evaluate_entity_extractors,
    evaluate_intent_classifiers,
    evaluate_language_detectors,
)
from .metrics import CombinedResults, EntityResults, IntentResults, LanguageResults
from .reporting import (
    export_results_json,
    print_table_combined,
    print_table_entity,
    print_table_entity_combined,
    print_table_intent,
    print_table_language,
)


def create_intent_classifiers(models: list[str]) -> list[tuple[str, Any]]:
    """Create intent classifiers based on model list."""
    classifiers: list[tuple[str, Any]] = []

    if "regex" in models or "all" in models:
        from src.nlp.intent import RegexIntentClassifier

        classifiers.append(("Regex", RegexIntentClassifier()))

    if "camembert" in models or "all" in models:
        from src.nlp.intent import CamembertIntentClassifier

        classifiers.append(("CamemBERT", CamembertIntentClassifier()))

    return classifiers


def create_entity_extractors(models: list[str]) -> list[tuple[str, Any]]:
    """Create entity extractors based on model list."""
    extractors: list[tuple[str, Any]] = []

    if "regex" in models or "all" in models:
        from src.nlp.entity import RegexEntityExtractor

        extractors.append(("Regex", RegexEntityExtractor()))

    if "spacy" in models or "all" in models:
        from src.nlp.entity import SpacyEntityExtractor

        extractors.append(("SpaCy", SpacyEntityExtractor()))

    if "camembert" in models or "all" in models:
        from src.nlp.entity import CamembertEntityExtractor

        extractors.append(("CamemBERT", CamembertEntityExtractor()))

    return extractors


def create_fuzzy_post_processor() -> Any:
    """Create fuzzy post-processor."""
    from src.nlp.post import FuzzyPostProcessor

    return FuzzyPostProcessor()


def create_language_detectors(models: list[str]) -> list[tuple[str, Any]]:
    """Create language detectors based on model list."""
    detectors: list[tuple[str, Any]] = []

    if "regex" in models or "all" in models:
        from src.nlp.language import RegexLanguageDetector

        detectors.append(("Regex", RegexLanguageDetector()))

    if "langdetect" in models or "all" in models:
        from src.nlp.language import LangdetectLanguageDetector

        detectors.append(("Langdetect", LangdetectLanguageDetector()))

    return detectors


def main() -> None:
    """Main entry point for evaluation CLI."""
    parser = argparse.ArgumentParser(description="Unified NLP evaluation for Travel Order Resolver")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to test dataset (JSON or CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/evaluation/reports",
        help="Directory for JSON output (default: src/evaluation/reports)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable automatic JSON export",
    )
    parser.add_argument(
        "--eval-type",
        choices=[
            "intent",
            "entity",
            "entity_fuzzy",
            "combined",
            "combined_fuzzy",
            "language",
            "all",
        ],
        default="all",
        help="Type of evaluation to run (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["regex", "spacy", "camembert", "langdetect", "all"],
        default=["all"],
        help="Models to evaluate (default: all)",
    )

    args = parser.parse_args()

    # Resolve dataset path
    dataset_path = args.dataset or str(
        Path(__file__).parent.parent.parent / "datasets" / "augmented" / "test.csv"
    )

    print("=" * 80)
    print("UNIFIED NLP EVALUATION - Travel Order Resolver")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    data = load_dataset(dataset_path)
    print(f"Loaded {len(data)} samples")

    # Count by intent
    intent_counts: dict[str, int] = {}
    for sample in data:
        intent = sample["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    print(f"Intent distribution: {intent_counts}")

    # Count by language
    language_counts: dict[str, int] = {}
    for sample in data:
        lang = sample.get("language", "UNKNOWN")
        language_counts[lang] = language_counts.get(lang, 0) + 1
    print(f"Language distribution: {language_counts}")

    # Initialize results
    intent_results: dict[str, IntentResults] = {}
    entity_results: dict[str, EntityResults] = {}
    entity_fuzzy_results: dict[str, EntityResults] = {}
    combined_results: list[CombinedResults] = []
    combined_fuzzy_results: list[CombinedResults] = []
    language_results: dict[str, LanguageResults] = {}

    eval_type = args.eval_type
    models = args.models

    # Load models
    print("\n" + "-" * 80)
    print("LOADING MODELS...")
    print("-" * 80)

    need_intent = eval_type in ["intent", "combined", "combined_fuzzy", "all"]
    need_entity = eval_type in ["entity", "entity_fuzzy", "combined", "combined_fuzzy", "all"]
    need_fuzzy = eval_type in ["entity_fuzzy", "combined_fuzzy", "all"]
    need_language = eval_type in ["language", "all"]

    classifiers = create_intent_classifiers(models) if need_intent else []
    extractors = create_entity_extractors(models) if need_entity else []
    fuzzy_post = create_fuzzy_post_processor() if need_fuzzy else None
    language_detectors = create_language_detectors(models) if need_language else []

    print("Models loaded successfully!")

    # Run evaluations
    if eval_type in ["intent", "all"]:
        print("\n" + "-" * 80)
        print("EVALUATING INTENT CLASSIFIERS...")
        print("-" * 80)
        if classifiers:
            intent_results = evaluate_intent_classifiers(classifiers, data)

    if eval_type in ["entity", "all"]:
        print("\n" + "-" * 80)
        print("EVALUATING ENTITY EXTRACTORS...")
        print("-" * 80)
        if extractors:
            entity_results = evaluate_entity_extractors(extractors, data)

    if eval_type in ["entity_fuzzy", "all"]:
        print("\n" + "-" * 80)
        print("EVALUATING ENTITY EXTRACTORS + FUZZY...")
        print("-" * 80)
        if extractors:
            entity_fuzzy_results = evaluate_entity_extractors(
                extractors, data, fuzzy_post=fuzzy_post, normalize_fuzzy=True
            )

    # Combined evaluations are slow - only run when explicitly requested
    if eval_type == "combined":
        print("\n" + "-" * 80)
        print("EVALUATING COMBINED (Intent x Entity)...")
        print("-" * 80)
        if classifiers and extractors:
            combined_results = evaluate_combined(classifiers, extractors, data)

    if eval_type == "combined_fuzzy":
        print("\n" + "-" * 80)
        print("EVALUATING COMBINED + FUZZY...")
        print("-" * 80)
        if classifiers and extractors:
            combined_fuzzy_results = evaluate_combined(
                classifiers, extractors, data, fuzzy_post=fuzzy_post, normalize_fuzzy=True
            )

    if eval_type in ["language", "all"]:
        print("\n" + "-" * 80)
        print("EVALUATING LANGUAGE DETECTORS...")
        print("-" * 80)
        if language_detectors:
            language_results = evaluate_language_detectors(language_detectors, data)

    # Print results
    if language_results:
        print_table_language(language_results)

    if intent_results:
        print_table_intent(intent_results)

    if entity_results and entity_fuzzy_results:
        print_table_entity_combined(entity_results, entity_fuzzy_results)
    elif entity_results:
        print_table_entity(entity_results, "ENTITY EXTRACTION", 3)
    elif entity_fuzzy_results:
        print_table_entity(entity_fuzzy_results, "ENTITY EXTRACTION + FUZZY", 3)

    if combined_fuzzy_results:
        print_table_combined(combined_fuzzy_results, "COMBINED PIPELINE", 4)

    # Export to JSON
    if not args.no_json:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = output_dir / f"evaluation_{timestamp}.json"

        export_results_json(
            intent_results,
            entity_results,
            entity_fuzzy_results,
            combined_results,
            combined_fuzzy_results,
            language_results,
            str(output_path),
        )
        print(f"\n\nResults exported to: {output_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
