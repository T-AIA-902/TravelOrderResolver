"""
Unified NLP evaluation script for Travel Order Resolver.

This script evaluates all available NLP methods using the modular architecture:
- Intent classifiers (Regex, CamemBERT)
- Entity extractors (Regex, SpaCy, CamemBERT)
- Post-processors (Fuzzy)

Produces 5 evaluation tables:
1. Intent Classification
2. Entity Extraction (without fuzzy)
3. Entity Extraction + Fuzzy
4. Combined (Intent x Entity)
5. Combined + Fuzzy (full pipeline)

Usage:
    python evaluation/evaluate_all.py --eval-type all
    python evaluation/evaluate_all.py --eval-type intent
    python evaluation/evaluate_all.py --eval-type entity
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Dataset Loading
# =============================================================================


def load_dataset(filepath: str) -> list:
    """
    Load dataset from JSON or CSV file.

    Returns normalized format with keys: sentence, intent, departure, destination
    """
    filepath = Path(filepath)

    if filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    elif filepath.suffix == ".csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = []
            for row in reader:
                sample = {
                    "sentence": row.get("text", row.get("sentence", "")),
                    "intent": map_label_to_intent(row.get("label", row.get("intent", ""))),
                    "departure": row.get("departure", ""),
                    "destination": row.get("destination", ""),
                    "intermediate": row.get("intermediate", ""),
                }
                data.append(sample)
        return data

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def map_label_to_intent(label: str) -> str:
    """Map CSV labels to standard intent format."""
    label = label.upper().strip()
    if label == "VALID":
        return "TRIP"
    elif label == "INVALID":
        return "NOT_TRIP"
    return label


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EntityMetrics:
    """Metrics for entity extraction (departure or destination)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class IntentResults:
    """Results for intent classification evaluation."""

    name: str
    correct: int = 0
    total: int = 0
    latencies: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)


@dataclass
class EntityResults:
    """Results for entity extraction evaluation."""

    name: str
    correct: int = 0
    total: int = 0
    departure_metrics: EntityMetrics = field(default_factory=EntityMetrics)
    destination_metrics: EntityMetrics = field(default_factory=EntityMetrics)
    latencies: list = field(default_factory=list)

    # Category metrics
    misspelling_correct: int = 0
    misspelling_total: int = 0
    lowercase_correct: int = 0
    lowercase_total: int = 0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def avg_precision(self) -> float:
        return (self.departure_metrics.precision + self.destination_metrics.precision) / 2

    @property
    def avg_recall(self) -> float:
        return (self.departure_metrics.recall + self.destination_metrics.recall) / 2

    @property
    def avg_f1(self) -> float:
        return (self.departure_metrics.f1_score + self.destination_metrics.f1_score) / 2


@dataclass
class CombinedResults:
    """Results for combined evaluation (intent + entity)."""

    intent_name: str
    entity_name: str
    with_fuzzy: bool = False

    intent_correct: int = 0
    intent_total: int = 0
    entity_correct: int = 0
    entity_total: int = 0
    latencies: list = field(default_factory=list)

    @property
    def name(self) -> str:
        suffix = " + Fuzzy" if self.with_fuzzy else ""
        return f"{self.intent_name} + {self.entity_name}{suffix}"

    @property
    def intent_accuracy(self) -> float:
        if self.intent_total == 0:
            return 0.0
        return self.intent_correct / self.intent_total

    @property
    def entity_accuracy(self) -> float:
        if self.entity_total == 0:
            return 0.0
        return self.entity_correct / self.entity_total

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)


# =============================================================================
# Helper Functions
# =============================================================================


def normalize_location(location: Optional[str]) -> Optional[str]:
    """Normalize location for comparison."""
    if location is None or location == "":
        return None
    return str(location).strip()


def normalize_fuzzy_station(station: Optional[str]) -> Optional[str]:
    """Extract city name from full SNCF station name."""
    if station is None or station == "":
        return None
    station = str(station).strip()
    if "-" in station:
        return station.split("-")[0]
    return station


def is_misspelled_sample(sentence: str, departure: str, destination: str) -> bool:
    """Detect if sentence contains misspelled station names."""
    sentence_lower = sentence.lower()
    if departure and departure.lower() not in sentence_lower:
        return True
    if destination and destination.lower() not in sentence_lower:
        return True
    return False


def is_lowercase_sample(sentence: str) -> bool:
    """Detect if sentence starts with lowercase."""
    if not sentence:
        return False
    for char in sentence:
        if char.isalpha():
            return char.islower()
    return False


def update_entity_metrics(
    metrics: EntityMetrics, predicted: Optional[str], ground_truth: Optional[str]
) -> bool:
    """Update entity metrics and return True if correct."""
    if predicted is not None and ground_truth is not None:
        if predicted == ground_truth:
            metrics.tp += 1
            return True
        else:
            metrics.fp += 1
            metrics.fn += 1
            return False
    elif predicted is not None and ground_truth is None:
        metrics.fp += 1
        return False
    elif predicted is None and ground_truth is not None:
        metrics.fn += 1
        return False
    else:
        metrics.tn += 1
        return True


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_intent_classifiers(
    classifiers: List[Tuple[str, Any]], data: list, batch_size: int = 128
) -> Dict[str, IntentResults]:
    """Evaluate all intent classifiers."""
    results = {}

    # Extract all sentences and true intents
    sentences = [sample["sentence"] for sample in data]
    true_intents = [sample["intent"] for sample in data]

    for name, classifier in classifiers:
        print(f"  Evaluating intent: {name}...")
        r = IntentResults(name=name)

        # Use batched classification if available (for CamemBERT)
        if hasattr(classifier, "classify_batch"):
            print(f"    Using batched inference (batch_size={batch_size})...")
            start = time.perf_counter()
            predictions = classifier.classify_batch(sentences, batch_size=batch_size)
            end = time.perf_counter()

            total_time_ms = (end - start) * 1000
            avg_latency = total_time_ms / len(sentences)

            for pred, true_intent in zip(predictions, true_intents):
                pred_intent, confidence = pred
                r.latencies.append(avg_latency)
                r.total += 1
                if pred_intent == true_intent:
                    r.correct += 1
        else:
            # Sequential classification for other classifiers
            for sentence, true_intent in zip(sentences, true_intents):
                start = time.perf_counter()
                pred_intent, confidence = classifier.classify(sentence)
                end = time.perf_counter()

                r.latencies.append((end - start) * 1000)
                r.total += 1

                if pred_intent == true_intent:
                    r.correct += 1

        results[name] = r
        print(f"    Done. Accuracy: {r.accuracy*100:.1f}%")

    return results


def evaluate_entity_extractors(
    extractors: List[Tuple[str, Any]],
    data: list,
    fuzzy_post: Any = None,
    normalize_fuzzy: bool = False,
    batch_size: int = 128,
) -> Dict[str, EntityResults]:
    """Evaluate all entity extractors."""
    results = {}

    # Filter TRIP samples once
    trip_samples = [s for s in data if s["intent"] == "TRIP"]
    sentences = [s["sentence"] for s in trip_samples]

    for name, extractor in extractors:
        suffix = " + Fuzzy" if fuzzy_post else ""
        display_name = f"{name}{suffix}"
        print(f"  Evaluating entity: {display_name}...")

        r = EntityResults(name=display_name)

        # Use batched extraction if available
        if hasattr(extractor, "extract_batch"):
            print(f"    Using batched inference (batch_size={batch_size})...")
            start = time.perf_counter()
            all_entities = extractor.extract_batch(sentences, batch_size=batch_size)
            end = time.perf_counter()

            total_time_ms = (end - start) * 1000
            avg_latency = total_time_ms / len(sentences) if sentences else 0

            for sample, entities in zip(trip_samples, all_entities):
                sentence = sample["sentence"]
                orig_true_dep = normalize_location(sample.get("departure"))
                orig_true_dest = normalize_location(sample.get("destination"))
                true_dep = orig_true_dep
                true_dest = orig_true_dest

                # Apply fuzzy post-processing if provided
                if fuzzy_post:
                    entities = fuzzy_post.process(entities, sentence)

                r.latencies.append(avg_latency)

                # Get predictions
                pred_dep = normalize_location(entities.get("departure"))
                pred_dest = normalize_location(entities.get("destination"))

                # Normalize for fuzzy comparison
                if normalize_fuzzy:
                    pred_dep = normalize_fuzzy_station(pred_dep)
                    pred_dest = normalize_fuzzy_station(pred_dest)
                    true_dep = normalize_fuzzy_station(true_dep)
                    true_dest = normalize_fuzzy_station(true_dest)

                r.total += 1

                # Update metrics
                dep_correct = update_entity_metrics(r.departure_metrics, pred_dep, true_dep)
                dest_correct = update_entity_metrics(r.destination_metrics, pred_dest, true_dest)

                both_correct = dep_correct and dest_correct
                if both_correct:
                    r.correct += 1

                # Category metrics
                if is_misspelled_sample(sentence, orig_true_dep or "", orig_true_dest or ""):
                    r.misspelling_total += 1
                    if both_correct:
                        r.misspelling_correct += 1

                if is_lowercase_sample(sentence):
                    r.lowercase_total += 1
                    if both_correct:
                        r.lowercase_correct += 1
        else:
            # Sequential extraction for extractors without batch support
            for sample in trip_samples:
                sentence = sample["sentence"]
                orig_true_dep = normalize_location(sample.get("departure"))
                orig_true_dest = normalize_location(sample.get("destination"))
                true_dep = orig_true_dep
                true_dest = orig_true_dest

                # Extract entities
                start = time.perf_counter()
                entities = extractor.extract(sentence)

                # Apply fuzzy post-processing if provided
                if fuzzy_post:
                    entities = fuzzy_post.process(entities, sentence)

                end = time.perf_counter()
                r.latencies.append((end - start) * 1000)

                # Get predictions
                pred_dep = normalize_location(entities.get("departure"))
                pred_dest = normalize_location(entities.get("destination"))

                # Normalize for fuzzy comparison
                if normalize_fuzzy:
                    pred_dep = normalize_fuzzy_station(pred_dep)
                    pred_dest = normalize_fuzzy_station(pred_dest)
                    true_dep = normalize_fuzzy_station(true_dep)
                    true_dest = normalize_fuzzy_station(true_dest)

                r.total += 1

                # Update metrics
                dep_correct = update_entity_metrics(r.departure_metrics, pred_dep, true_dep)
                dest_correct = update_entity_metrics(r.destination_metrics, pred_dest, true_dest)

                both_correct = dep_correct and dest_correct
                if both_correct:
                    r.correct += 1

                # Category metrics
                if is_misspelled_sample(sentence, orig_true_dep or "", orig_true_dest or ""):
                    r.misspelling_total += 1
                    if both_correct:
                        r.misspelling_correct += 1

                if is_lowercase_sample(sentence):
                    r.lowercase_total += 1
                    if both_correct:
                        r.lowercase_correct += 1

        results[display_name] = r
        print(f"    Done. Accuracy: {r.accuracy*100:.1f}%")

    return results


def evaluate_combined(
    intent_classifiers: List[Tuple[str, Any]],
    entity_extractors: List[Tuple[str, Any]],
    data: list,
    fuzzy_post: Any = None,
    normalize_fuzzy: bool = False,
) -> List[CombinedResults]:
    """Evaluate all combinations of intent classifiers and entity extractors."""
    results = []

    for intent_name, intent_clf in intent_classifiers:
        for entity_name, entity_ext in entity_extractors:
            suffix = " + Fuzzy" if fuzzy_post else ""
            print(f"  Evaluating combined: {intent_name} + {entity_name}{suffix}...")

            r = CombinedResults(
                intent_name=intent_name,
                entity_name=entity_name,
                with_fuzzy=fuzzy_post is not None,
            )

            for sample in data:
                sentence = sample["sentence"]
                true_intent = sample["intent"]

                # Get ground truth
                orig_true_dep = normalize_location(sample.get("departure"))
                orig_true_dest = normalize_location(sample.get("destination"))
                true_dep = orig_true_dep
                true_dest = orig_true_dest

                start = time.perf_counter()

                # Classify intent
                pred_intent, _ = intent_clf.classify(sentence)

                # Extract entities
                entities = entity_ext.extract(sentence)
                if fuzzy_post:
                    entities = fuzzy_post.process(entities, sentence)

                end = time.perf_counter()
                r.latencies.append((end - start) * 1000)

                # Intent evaluation
                r.intent_total += 1
                if pred_intent == true_intent:
                    r.intent_correct += 1

                # Entity evaluation (only on TRIP samples)
                if true_intent == "TRIP":
                    pred_dep = normalize_location(entities.get("departure"))
                    pred_dest = normalize_location(entities.get("destination"))

                    if normalize_fuzzy:
                        pred_dep = normalize_fuzzy_station(pred_dep)
                        pred_dest = normalize_fuzzy_station(pred_dest)
                        true_dep = normalize_fuzzy_station(true_dep)
                        true_dest = normalize_fuzzy_station(true_dest)

                    r.entity_total += 1
                    if pred_dep == true_dep and pred_dest == true_dest:
                        r.entity_correct += 1

            results.append(r)
            print(
                f"    Done. Intent: {r.intent_accuracy*100:.1f}%, "
                f"Entity: {r.entity_accuracy*100:.1f}%"
            )

    return results


# =============================================================================
# Reporting Functions
# =============================================================================


def print_table_intent(results: Dict[str, IntentResults]):
    """Print Table 1: Intent Classification."""
    print("\n" + "=" * 80)
    print("TABLE 1: INTENT CLASSIFICATION")
    print("=" * 80)
    print("\n| Model | Accuracy | Latency |")
    print("|-------|----------|---------|")

    for name, r in results.items():
        print(f"| {name:<15} | {r.accuracy*100:>7.1f}% | {r.avg_latency_ms:>6.1f}ms |")


def print_table_entity(results: Dict[str, EntityResults], title: str, table_num: int):
    """Print entity extraction table."""
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


def print_table_combined(results: List[CombinedResults], title: str, table_num: int):
    """Print combined evaluation table."""
    print("\n" + "=" * 80)
    print(f"TABLE {table_num}: {title}")
    print("=" * 80)
    print("\n| Intent | Entity | Intent Acc | Entity Acc | Latency |")
    print("|--------|--------|------------|------------|---------|")

    for r in results:
        suffix = " + Fuzzy" if r.with_fuzzy else ""
        print(
            f"| {r.intent_name:<10} | {r.entity_name + suffix:<18} | "
            f"{r.intent_accuracy*100:>9.1f}% | {r.entity_accuracy*100:>9.1f}% | "
            f"{r.avg_latency_ms:>6.1f}ms |"
        )


def export_results_json(
    intent_results: Dict[str, IntentResults],
    entity_results: Dict[str, EntityResults],
    entity_fuzzy_results: Dict[str, EntityResults],
    combined_results: List[CombinedResults],
    combined_fuzzy_results: List[CombinedResults],
    output_path: str,
):
    """Export all results to JSON file."""
    export_data = {
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
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)


# =============================================================================
# Model Factory
# =============================================================================


def create_intent_classifiers(models: List[str]) -> List[Tuple[str, Any]]:
    """Create intent classifiers based on model list."""
    classifiers = []

    if "regex" in models or "all" in models:
        from src.nlp.intent import RegexIntentClassifier

        classifiers.append(("Regex", RegexIntentClassifier()))

    if "camembert" in models or "all" in models:
        from src.nlp.intent import CamembertIntentClassifier

        classifiers.append(("CamemBERT", CamembertIntentClassifier()))

    if "flant5" in models or "all" in models:
        from src.nlp.intent import FlanT5IntentClassifier

        classifiers.append(("Flan-T5", FlanT5IntentClassifier()))

    return classifiers


def create_entity_extractors(models: List[str]) -> List[Tuple[str, Any]]:
    """Create entity extractors based on model list."""
    extractors = []

    if "regex" in models or "all" in models:
        from src.nlp.entity import RegexEntityExtractor

        extractors.append(("Regex", RegexEntityExtractor()))

    if "spacy" in models or "all" in models:
        from src.nlp.entity import SpacyEntityExtractor

        extractors.append(("SpaCy", SpacyEntityExtractor()))

    if "camembert" in models or "all" in models:
        from src.nlp.entity import CamembertEntityExtractor

        extractors.append(("CamemBERT", CamembertEntityExtractor()))

    if "flant5" in models or "all" in models:
        from src.nlp.entity import FlanT5EntityExtractor

        extractors.append(("Flan-T5", FlanT5EntityExtractor()))

    return extractors


def create_fuzzy_post_processor() -> Any:
    """Create fuzzy post-processor."""
    from src.nlp.post import FuzzyPostProcessor

    return FuzzyPostProcessor()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Unified NLP evaluation for Travel Order Resolver")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to test dataset (JSON or CSV)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--eval-type",
        choices=["intent", "entity", "entity_fuzzy", "combined", "combined_fuzzy", "all"],
        default="all",
        help="Type of evaluation to run (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["regex", "spacy", "camembert", "flant5", "all"],
        default=["all"],
        help="Models to evaluate (default: all)",
    )

    args = parser.parse_args()

    # Resolve dataset path
    dataset_path = args.dataset or (
        Path(__file__).parent.parent / "datasets" / "splits" / "test.csv"
    )

    print("=" * 80)
    print("UNIFIED NLP EVALUATION - Travel Order Resolver (Refactored)")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    data = load_dataset(str(dataset_path))
    print(f"Loaded {len(data)} samples")

    # Count by intent
    intent_counts: Dict[str, int] = {}
    for sample in data:
        intent = sample["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    print(f"Intent distribution: {intent_counts}")

    # Initialize results
    intent_results: Dict[str, IntentResults] = {}
    entity_results: Dict[str, EntityResults] = {}
    entity_fuzzy_results: Dict[str, EntityResults] = {}
    combined_results: List[CombinedResults] = []
    combined_fuzzy_results: List[CombinedResults] = []

    eval_type = args.eval_type
    models = args.models

    # ==========================================================================
    # Create models ONCE and reuse across all evaluations
    # ==========================================================================
    print("\n" + "-" * 80)
    print("LOADING MODELS...")
    print("-" * 80)

    # Determine which models we need
    need_intent = eval_type in ["intent", "combined", "combined_fuzzy", "all"]
    need_entity = eval_type in ["entity", "entity_fuzzy", "combined", "combined_fuzzy", "all"]
    need_fuzzy = eval_type in ["entity_fuzzy", "combined_fuzzy", "all"]

    classifiers = create_intent_classifiers(models) if need_intent else []
    extractors = create_entity_extractors(models) if need_entity else []
    fuzzy_post = create_fuzzy_post_processor() if need_fuzzy else None

    print("Models loaded successfully!")

    # ==========================================================================
    # Table 1: Intent Classification
    # ==========================================================================
    if eval_type in ["intent", "all"]:
        print("\n" + "-" * 80)
        print("EVALUATING INTENT CLASSIFIERS...")
        print("-" * 80)

        if classifiers:
            intent_results = evaluate_intent_classifiers(classifiers, data)

    # ==========================================================================
    # Table 2: Entity Extraction (without fuzzy)
    # ==========================================================================
    if eval_type in ["entity", "all"]:
        print("\n" + "-" * 80)
        print("EVALUATING ENTITY EXTRACTORS...")
        print("-" * 80)

        if extractors:
            entity_results = evaluate_entity_extractors(extractors, data)

    # ==========================================================================
    # Table 3: Entity Extraction + Fuzzy
    # ==========================================================================
    if eval_type in ["entity_fuzzy", "all"]:
        print("\n" + "-" * 80)
        print("EVALUATING ENTITY EXTRACTORS + FUZZY...")
        print("-" * 80)

        if extractors:
            entity_fuzzy_results = evaluate_entity_extractors(
                extractors, data, fuzzy_post=fuzzy_post, normalize_fuzzy=True
            )

    # ==========================================================================
    # Table 4: Combined (Intent x Entity)
    # ==========================================================================
    if eval_type in ["combined", "all"]:
        print("\n" + "-" * 80)
        print("EVALUATING COMBINED (Intent x Entity)...")
        print("-" * 80)

        if classifiers and extractors:
            combined_results = evaluate_combined(classifiers, extractors, data)

    # ==========================================================================
    # Table 5: Combined + Fuzzy
    # ==========================================================================
    if eval_type in ["combined_fuzzy", "all"]:
        print("\n" + "-" * 80)
        print("EVALUATING COMBINED + FUZZY...")
        print("-" * 80)

        if classifiers and extractors:
            combined_fuzzy_results = evaluate_combined(
                classifiers, extractors, data, fuzzy_post=fuzzy_post, normalize_fuzzy=True
            )

    # ==========================================================================
    # Print Results
    # ==========================================================================
    if intent_results:
        print_table_intent(intent_results)

    if entity_results:
        print_table_entity(entity_results, "ENTITY EXTRACTION (without fuzzy)", 2)

    if entity_fuzzy_results:
        print_table_entity(entity_fuzzy_results, "ENTITY EXTRACTION + FUZZY", 3)

    if combined_results:
        print_table_combined(combined_results, "COMBINED (Intent x Entity)", 4)

    if combined_fuzzy_results:
        print_table_combined(combined_fuzzy_results, "COMBINED + FUZZY", 5)

    # Export to JSON if requested
    if args.output_json:
        export_results_json(
            intent_results,
            entity_results,
            entity_fuzzy_results,
            combined_results,
            combined_fuzzy_results,
            args.output_json,
        )
        print(f"\n\nResults exported to: {args.output_json}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
