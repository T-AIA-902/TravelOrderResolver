"""
Unified NLP evaluation script for Travel Order Resolver.

This script evaluates all available NLP methods (regex, spacy, fuzzy)
and generates comprehensive metrics tables for the README.

Usage:
    python evaluation/evaluate_all.py
    python evaluation/evaluate_all.py --models spacy fuzzy
    python evaluation/evaluate_all.py --output-json results.json
"""

import argparse
import csv
import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import StationDatabase  # noqa: E402
from src.nlp.entity_extractor import FuzzyEntityExtractor, SpacyEntityExtractor  # noqa: E402
from src.nlp.models.baseline_regex import BaselineRegexModel  # noqa: E402

# =============================================================================
# Dataset Loading
# =============================================================================


def load_dataset(filepath: str) -> list:
    """
    Load dataset from JSON or CSV file.

    Handles two formats:
    - JSON (test.json): {"sentence", "intent", "departure", "destination"}
    - CSV (splits/*.csv): {"text", "label", "departure", "destination"}

    Returns normalized format with keys: sentence, intent, departure, destination
    """
    filepath = Path(filepath)

    if filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    elif filepath.suffix == ".csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = []
            for row in reader:
                # Map CSV format to standard format
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
    """
    Map CSV labels to standard intent format.

    VALID -> TRIP
    INVALID -> NOT_TRIP
    """
    label = label.upper().strip()
    if label == "VALID":
        return "TRIP"
    elif label == "INVALID":
        return "NOT_TRIP"
    else:
        # Already in standard format
        return label


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EntityMetrics:
    """Metrics for entity extraction (departure or destination)."""

    tp: int = 0  # True positives
    fp: int = 0  # False positives
    fn: int = 0  # False negatives
    tn: int = 0  # True negatives

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
class EvaluationResults:
    """Complete evaluation results for a single model."""

    model_name: str

    # Intent metrics
    intent_correct: int = 0
    intent_total: int = 0

    # Entity metrics (both correct)
    entity_correct: int = 0
    entity_total: int = 0

    # Per-entity metrics
    departure_metrics: EntityMetrics = field(default_factory=EntityMetrics)
    destination_metrics: EntityMetrics = field(default_factory=EntityMetrics)

    # Category-specific metrics
    misspelling_correct: int = 0
    misspelling_total: int = 0
    lowercase_correct: int = 0
    lowercase_total: int = 0
    order_correct: int = 0
    order_total: int = 0

    # Latency
    latencies: list = field(default_factory=list)

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

    @property
    def misspelling_accuracy(self) -> float:
        if self.misspelling_total == 0:
            return 0.0
        return self.misspelling_correct / self.misspelling_total

    @property
    def lowercase_accuracy(self) -> float:
        if self.lowercase_total == 0:
            return 0.0
        return self.lowercase_correct / self.lowercase_total

    @property
    def order_accuracy(self) -> float:
        if self.order_total == 0:
            return 0.0
        return self.order_correct / self.order_total


# =============================================================================
# Model Adapters
# =============================================================================


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> dict:
        """Extract entities from text. Returns dict with departure, destination."""
        pass

    def classify_intent(self, text: str) -> Optional[str]:
        """Classify intent. Returns None if not supported."""
        return None


class SpacyAdapter(ModelAdapter):
    """Adapter for SpaCy entity extractor (no intent classification)."""

    def __init__(self):
        self._model = SpacyEntityExtractor()

    @property
    def name(self) -> str:
        return "SpaCy"

    def extract_entities(self, text: str) -> dict:
        return self._model.extract_entities(text)


class FuzzyAdapter(ModelAdapter):
    """Adapter for Fuzzy entity extractor (no intent classification)."""

    def __init__(self, threshold: int = 75):
        self._model = FuzzyEntityExtractor(fuzzy_threshold=threshold)

    @property
    def name(self) -> str:
        return "Fuzzy"

    def extract_entities(self, text: str) -> dict:
        return self._model.extract_entities(text)


class RegexAdapter(ModelAdapter):
    """Adapter for Baseline Regex model (intent + entities)."""

    def __init__(self):
        station_db = StationDatabase()
        station_db.load()
        self._model = BaselineRegexModel(station_db=station_db)

    @property
    def name(self) -> str:
        return "Baseline Regex"

    def extract_entities(self, text: str) -> dict:
        result = self._model.predict(text)
        return {
            "departure": result.departure or None,
            "destination": result.destination or None,
            "intermediate": result.intermediates or [],
        }

    def classify_intent(self, text: str) -> Optional[str]:
        result = self._model.predict(text)
        return result.intent.value


class SpacyRegexAdapter(ModelAdapter):
    """Adapter combining Regex (intent) + SpaCy (entities)."""

    def __init__(self):
        station_db = StationDatabase()
        station_db.load()
        self._regex_model = BaselineRegexModel(station_db=station_db)
        self._spacy_model = SpacyEntityExtractor()

    @property
    def name(self) -> str:
        return "SpaCy + Regex"

    def extract_entities(self, text: str) -> dict:
        return self._spacy_model.extract_entities(text)

    def classify_intent(self, text: str) -> Optional[str]:
        result = self._regex_model.predict(text)
        return result.intent.value


class FuzzyRegexAdapter(ModelAdapter):
    """Adapter combining Regex (intent) + Fuzzy (entities)."""

    def __init__(self, threshold: int = 75):
        station_db = StationDatabase()
        station_db.load()
        self._regex_model = BaselineRegexModel(station_db=station_db)
        self._fuzzy_model = FuzzyEntityExtractor(fuzzy_threshold=threshold)

    @property
    def name(self) -> str:
        return "Fuzzy + Regex"

    def extract_entities(self, text: str) -> dict:
        return self._fuzzy_model.extract_entities(text)

    def classify_intent(self, text: str) -> Optional[str]:
        result = self._regex_model.predict(text)
        return result.intent.value


# =============================================================================
# Helper Functions
# =============================================================================


def normalize_location(location: Optional[str]) -> Optional[str]:
    """Normalize location for comparison."""
    if location is None or location == "":
        return None
    return str(location).strip()


def normalize_fuzzy_station(station: Optional[str]) -> Optional[str]:
    """
    Extract city name from full SNCF station name.
    'Paris-Gare-de-Lyon' -> 'Paris'
    'Lyon-Part-Dieu' -> 'Lyon'
    """
    if station is None or station == "":
        return None
    station = str(station).strip()
    if "-" in station:
        return station.split("-")[0]
    return station


def is_misspelled_sample(sentence: str, departure: str, destination: str) -> bool:
    """
    Detect if the sentence contains misspelled station names.
    If the ground truth doesn't appear exactly in the sentence,
    the sentence contains a misspelling.
    """
    sentence_lower = sentence.lower()

    if departure:
        dep_lower = departure.lower()
        if dep_lower not in sentence_lower:
            return True

    if destination:
        dest_lower = destination.lower()
        if dest_lower not in sentence_lower:
            return True

    return False


def is_lowercase_sample(sentence: str) -> bool:
    """Detect if the sentence starts with a lowercase letter."""
    if not sentence:
        return False
    for char in sentence:
        if char.isalpha():
            return char.islower()
    return False


def get_order_from_sentence(sentence: str, departure: str, destination: str) -> Optional[tuple]:
    """
    Get the order of departure and destination in the sentence.
    Returns (dep_pos, dest_pos) or None if not found.
    """
    sentence_lower = sentence.lower()
    dep_pos = sentence_lower.find(departure.lower()) if departure else -1
    dest_pos = sentence_lower.find(destination.lower()) if destination else -1

    if dep_pos >= 0 and dest_pos >= 0:
        return (dep_pos, dest_pos)
    return None


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
    else:  # Both None
        metrics.tn += 1
        return True


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_model(
    adapter: ModelAdapter,
    data: list,
    include_intent: bool = False,
    normalize_fuzzy: bool = False,
) -> EvaluationResults:
    """
    Evaluate a model on the test dataset.

    Args:
        adapter: Model adapter to evaluate
        data: List of test samples (dicts)
        include_intent: Whether to evaluate intent classification
        normalize_fuzzy: Whether to normalize fuzzy station names (both pred and ground truth)

    Returns:
        EvaluationResults with all metrics
    """
    results = EvaluationResults(model_name=adapter.name)

    for sample in data:
        sentence = sample["sentence"]
        true_intent = sample["intent"]

        # Keep original ground truth for category detection
        orig_true_departure = normalize_location(sample.get("departure"))
        orig_true_destination = normalize_location(sample.get("destination"))

        # Values used for comparison (may be normalized for fuzzy)
        true_departure = orig_true_departure
        true_destination = orig_true_destination

        # Measure latency
        start_time = time.perf_counter()
        entities = adapter.extract_entities(sentence)
        intent_pred = adapter.classify_intent(sentence) if include_intent else None
        end_time = time.perf_counter()
        results.latencies.append((end_time - start_time) * 1000)

        # Get predicted entities
        pred_departure = normalize_location(entities.get("departure"))
        pred_destination = normalize_location(entities.get("destination"))

        # Normalize for fuzzy comparison (both predictions AND ground truth)
        if normalize_fuzzy:
            pred_departure = normalize_fuzzy_station(pred_departure)
            pred_destination = normalize_fuzzy_station(pred_destination)
            true_departure = normalize_fuzzy_station(true_departure)
            true_destination = normalize_fuzzy_station(true_destination)

        # Evaluate intent (if applicable)
        if include_intent and intent_pred is not None:
            results.intent_total += 1
            if intent_pred == true_intent:
                results.intent_correct += 1

        # Only evaluate entity extraction on TRIP sentences
        if true_intent != "TRIP":
            continue

        results.entity_total += 1

        # Update departure metrics
        dep_correct = update_entity_metrics(
            results.departure_metrics, pred_departure, true_departure
        )

        # Update destination metrics
        dest_correct = update_entity_metrics(
            results.destination_metrics, pred_destination, true_destination
        )

        # Both correct?
        both_correct = dep_correct and dest_correct
        if both_correct:
            results.entity_correct += 1

        # Category: Misspelling handling (use original ground truth)
        if is_misspelled_sample(sentence, orig_true_departure or "", orig_true_destination or ""):
            results.misspelling_total += 1
            if both_correct:
                results.misspelling_correct += 1

        # Category: Lowercase handling
        if is_lowercase_sample(sentence):
            results.lowercase_total += 1
            if both_correct:
                results.lowercase_correct += 1

        # Category: Order detection (use original ground truth)
        if orig_true_departure and orig_true_destination:
            order = get_order_from_sentence(sentence, orig_true_departure, orig_true_destination)
            if order is not None:
                dep_pos, dest_pos = order
                results.order_total += 1
                # Check if model got the order right
                if both_correct:
                    results.order_correct += 1
                elif pred_departure == true_destination and pred_destination == true_departure:
                    # Model swapped departure and destination
                    pass  # Already counted as wrong
                elif dep_correct or dest_correct:
                    # Partially correct
                    results.order_correct += 1

    return results


# =============================================================================
# Reporting Functions
# =============================================================================


def print_table_1_extractors(results: dict):
    """Print Table 1: Entity extractors only (SpaCy, Fuzzy)."""
    print("\n" + "=" * 80)
    print("TABLE 1: EXTRACTEURS D'ENTITES (Entity Extraction Only)")
    print("=" * 80)
    print("\n| Methode | Accuracy | Precision | Recall | F1-Score | Latence |")
    print("|---------|----------|-----------|--------|----------|---------|")

    for name in ["SpaCy", "Fuzzy"]:
        if name in results:
            r = results[name]
            dep = r.departure_metrics
            dest = r.destination_metrics
            # Average precision/recall/f1 for departure and destination
            avg_precision = (dep.precision + dest.precision) / 2
            avg_recall = (dep.recall + dest.recall) / 2
            avg_f1 = (dep.f1_score + dest.f1_score) / 2
            print(
                f"| {name:<7} | {r.entity_accuracy*100:>7.1f}% | "
                f"{avg_precision*100:>8.1f}% | {avg_recall*100:>5.1f}% | "
                f"{avg_f1*100:>7.1f}% | {r.avg_latency_ms:>6.1f}ms |"
            )


def print_table_2_complete(results: dict):
    """Print Table 2: Complete solutions (Intent + Entities)."""
    print("\n" + "=" * 80)
    print("TABLE 2: SOLUTIONS COMPLETES (Intent + Entity Extraction)")
    print("=" * 80)
    print("\n| Methode | Intent Acc | Entity Acc | Precision | Recall | F1-Score | Latence |")
    print("|---------|------------|------------|-----------|--------|----------|---------|")

    for name in ["Baseline Regex", "SpaCy + Regex", "Fuzzy + Regex"]:
        if name in results:
            r = results[name]
            dep = r.departure_metrics
            dest = r.destination_metrics
            avg_precision = (dep.precision + dest.precision) / 2
            avg_recall = (dep.recall + dest.recall) / 2
            avg_f1 = (dep.f1_score + dest.f1_score) / 2
            print(
                f"| {name:<13} | {r.intent_accuracy*100:>9.1f}% | "
                f"{r.entity_accuracy*100:>9.1f}% | {avg_precision*100:>8.1f}% | "
                f"{avg_recall*100:>5.1f}% | {avg_f1*100:>7.1f}% | {r.avg_latency_ms:>6.1f}ms |"
            )


def print_table_3_categories(results: dict):
    """Print Table 3: Category-specific metrics."""
    print("\n" + "=" * 80)
    print("TABLE 3: METRIQUES PAR CATEGORIE")
    print("=" * 80)

    # Get complete solution names
    complete_names = ["Baseline Regex", "SpaCy + Regex", "Fuzzy + Regex"]
    available = [n for n in complete_names if n in results]

    if not available:
        print("No complete solutions evaluated.")
        return

    # Header
    header = "| Categorie |"
    for name in available:
        header += f" {name:<13} |"
    print(f"\n{header}")

    divider = "|-----------|"
    for _ in available:
        divider += "---------------|"
    print(divider)

    # Departure Precision
    row = "| Dep. Precision |"
    for name in available:
        r = results[name]
        row += f" {r.departure_metrics.precision*100:>12.1f}% |"
    print(row)

    # Departure Recall
    row = "| Dep. Recall |"
    for name in available:
        r = results[name]
        row += f" {r.departure_metrics.recall*100:>12.1f}% |"
    print(row)

    # Departure F1
    row = "| Dep. F1 |"
    for name in available:
        r = results[name]
        row += f" {r.departure_metrics.f1_score*100:>12.1f}% |"
    print(row)

    # Destination Precision
    row = "| Dest. Precision |"
    for name in available:
        r = results[name]
        row += f" {r.destination_metrics.precision*100:>12.1f}% |"
    print(row)

    # Destination Recall
    row = "| Dest. Recall |"
    for name in available:
        r = results[name]
        row += f" {r.destination_metrics.recall*100:>12.1f}% |"
    print(row)

    # Destination F1
    row = "| Dest. F1 |"
    for name in available:
        r = results[name]
        row += f" {r.destination_metrics.f1_score*100:>12.1f}% |"
    print(row)

    # Order Detection
    row = "| Dep/Dest Order |"
    for name in available:
        r = results[name]
        if r.order_total > 0:
            row += f" {r.order_accuracy*100:>11.1f}% |"
        else:
            row += f" {'N/A':>12} |"
    print(row)

    # Misspelling Handling
    row = "| Misspelling |"
    for name in available:
        r = results[name]
        if r.misspelling_total > 0:
            row += f" {r.misspelling_accuracy*100:>5.1f}% ({r.misspelling_total:>3}) |"
        else:
            row += f" {'N/A':>12} |"
    print(row)

    # Lowercase Handling
    row = "| No-caps |"
    for name in available:
        r = results[name]
        if r.lowercase_total > 0:
            row += f" {r.lowercase_accuracy*100:>5.1f}% ({r.lowercase_total:>3}) |"
        else:
            row += f" {'N/A':>12} |"
    print(row)


def export_results_json(results: dict, output_path: str):
    """Export results to JSON file."""
    export_data = {}
    for name, r in results.items():
        export_data[name] = {
            "intent_accuracy": r.intent_accuracy,
            "entity_accuracy": r.entity_accuracy,
            "departure_precision": r.departure_metrics.precision,
            "departure_recall": r.departure_metrics.recall,
            "departure_f1": r.departure_metrics.f1_score,
            "destination_precision": r.destination_metrics.precision,
            "destination_recall": r.destination_metrics.recall,
            "destination_f1": r.destination_metrics.f1_score,
            "avg_latency_ms": r.avg_latency_ms,
            "misspelling_accuracy": r.misspelling_accuracy,
            "misspelling_total": r.misspelling_total,
            "lowercase_accuracy": r.lowercase_accuracy,
            "lowercase_total": r.lowercase_total,
            "order_accuracy": r.order_accuracy,
            "order_total": r.order_total,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Unified NLP evaluation for Travel Order Resolver")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to test dataset (JSON or CSV). Default: datasets/generated/test.json",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Export results to JSON file")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["spacy", "fuzzy", "regex", "spacy_regex", "fuzzy_regex", "all"],
        default=["all"],
        help="Models to evaluate (default: all)",
    )

    args = parser.parse_args()

    # Resolve dataset path
    dataset_path = args.dataset or (
        Path(__file__).parent.parent / "datasets" / "generated" / "test.json"
    )

    print("=" * 80)
    print("UNIFIED NLP EVALUATION - Travel Order Resolver")
    print("=" * 80)

    # Load dataset (supports both JSON and CSV)
    print(f"\nLoading dataset from: {dataset_path}")
    data = load_dataset(str(dataset_path))
    print(f"Loaded {len(data)} samples")

    # Count by intent
    intent_counts = {}
    for sample in data:
        intent = sample["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    print(f"Intent distribution: {intent_counts}")

    # Determine which models to evaluate
    if "all" in args.models:
        model_list = ["spacy", "fuzzy", "regex", "spacy_regex", "fuzzy_regex"]
    else:
        model_list = args.models

    results = {}

    # Initialize and evaluate each model
    print("\n" + "-" * 80)
    print("EVALUATING MODELS...")
    print("-" * 80)

    # Entity extractors (no intent)
    if "spacy" in model_list:
        print("\n[1/5] Initializing SpaCy...")
        adapter = SpacyAdapter()
        print("      Evaluating SpaCy...")
        results["SpaCy"] = evaluate_model(adapter, data, include_intent=False)
        print(f"      Done. Entity accuracy: {results['SpaCy'].entity_accuracy*100:.1f}%")

    if "fuzzy" in model_list:
        print("\n[2/5] Initializing Fuzzy...")
        adapter = FuzzyAdapter()
        print("      Evaluating Fuzzy...")
        results["Fuzzy"] = evaluate_model(adapter, data, include_intent=False, normalize_fuzzy=True)
        print(f"      Done. Entity accuracy: {results['Fuzzy'].entity_accuracy*100:.1f}%")

    # Complete solutions (intent + entities)
    if "regex" in model_list:
        print("\n[3/5] Initializing Baseline Regex...")
        adapter = RegexAdapter()
        print("      Evaluating Baseline Regex...")
        results["Baseline Regex"] = evaluate_model(adapter, data, include_intent=True)
        print(
            f"      Done. Intent: {results['Baseline Regex'].intent_accuracy*100:.1f}%, "
            f"Entity: {results['Baseline Regex'].entity_accuracy*100:.1f}%"
        )

    if "spacy_regex" in model_list:
        print("\n[4/5] Initializing SpaCy + Regex...")
        adapter = SpacyRegexAdapter()
        print("      Evaluating SpaCy + Regex...")
        results["SpaCy + Regex"] = evaluate_model(adapter, data, include_intent=True)
        print(
            f"      Done. Intent: {results['SpaCy + Regex'].intent_accuracy*100:.1f}%, "
            f"Entity: {results['SpaCy + Regex'].entity_accuracy*100:.1f}%"
        )

    if "fuzzy_regex" in model_list:
        print("\n[5/5] Initializing Fuzzy + Regex...")
        adapter = FuzzyRegexAdapter()
        print("      Evaluating Fuzzy + Regex...")
        results["Fuzzy + Regex"] = evaluate_model(
            adapter, data, include_intent=True, normalize_fuzzy=True
        )
        print(
            f"      Done. Intent: {results['Fuzzy + Regex'].intent_accuracy*100:.1f}%, "
            f"Entity: {results['Fuzzy + Regex'].entity_accuracy*100:.1f}%"
        )

    # Print results tables
    print_table_1_extractors(results)
    print_table_2_complete(results)
    print_table_3_categories(results)

    # Export to JSON if requested
    if args.output_json:
        export_results_json(results, args.output_json)
        print(f"\n\nResults exported to: {args.output_json}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
