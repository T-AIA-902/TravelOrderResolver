"""
Metrics dataclasses for evaluation results.

Contains result containers for different evaluation types.
"""

from dataclasses import dataclass, field


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
class IntentResults:
    """Results for intent classification evaluation."""

    name: str
    correct: int = 0
    total: int = 0
    latencies: list[float] = field(default_factory=list)

    # Per-language breakdown
    french_correct: int = 0
    french_total: int = 0
    english_correct: int = 0
    english_total: int = 0
    unknown_correct: int = 0
    unknown_total: int = 0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def french_accuracy(self) -> float:
        if self.french_total == 0:
            return 0.0
        return self.french_correct / self.french_total

    @property
    def english_accuracy(self) -> float:
        if self.english_total == 0:
            return 0.0
        return self.english_correct / self.english_total

    @property
    def unknown_accuracy(self) -> float:
        if self.unknown_total == 0:
            return 0.0
        return self.unknown_correct / self.unknown_total

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
    latencies: list[float] = field(default_factory=list)

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
    latencies: list[float] = field(default_factory=list)

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


@dataclass
class LanguageResults:
    """Results for language detection evaluation."""

    name: str
    total: int = 0
    correct: int = 0
    french_correct: int = 0
    french_total: int = 0
    english_correct: int = 0
    english_total: int = 0
    unknown_correct: int = 0
    unknown_total: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def french_accuracy(self) -> float:
        if self.french_total == 0:
            return 0.0
        return self.french_correct / self.french_total

    @property
    def english_accuracy(self) -> float:
        if self.english_total == 0:
            return 0.0
        return self.english_correct / self.english_total

    @property
    def unknown_accuracy(self) -> float:
        if self.unknown_total == 0:
            return 0.0
        return self.unknown_correct / self.unknown_total

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)


def update_entity_metrics(
    metrics: EntityMetrics, predicted: str | None, ground_truth: str | None
) -> bool:
    """
    Update entity metrics and return True if correct.

    Args:
        metrics: EntityMetrics instance to update
        predicted: Predicted value (or None)
        ground_truth: Ground truth value (or None)

    Returns:
        True if prediction matches ground truth
    """
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
