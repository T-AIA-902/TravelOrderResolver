"""
CamemBERT NER-derived intent classifier.

Derives intent from fine-tuned NER results: if DEP/DEST entities
are found the sentence is classified as TRIP, otherwise NOT_TRIP.
Shares the NER model with CamembertEntityExtractor to avoid
loading the 420 MB model twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Literal, Tuple

from ..interfaces import IntentClassifier

DeviceType = Literal["auto", "cuda", "cpu"]


class CamembertIntentClassifier(IntentClassifier):
    """NER-derived intent classifier using fine-tuned CamemBERT.

    Derives intent from NER results:
    - DEP + DEST found  -> TRIP  (confidence 0.9)
    - DEP or DEST found -> TRIP  (confidence 0.7)
    - Nothing found     -> NOT_TRIP (confidence 0.8)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: DeviceType = "auto",
        ner_model: Any | None = None,
    ) -> None:
        """Initialize the CamemBERT intent classifier.

        Args:
            model_path: Path to the fine-tuned model directory.
                Defaults to models/camembert-ner-retrain/.
            device: Device to use - "auto", "cuda", or "cpu".
            ner_model: Optional shared CamembertNERModel instance.
                If provided, reuses this model instead of loading a new one.
        """
        if ner_model is not None:
            self.ner_model = ner_model
        else:
            from src.nlp.camembert_ner_model import CamembertNERModel

            self.ner_model = CamembertNERModel(model_path=model_path, device=device)

    @property
    def name(self) -> str:
        """Return the classifier name."""
        return "CamemBERT"

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify the intent of the input text via NER.

        Args:
            text: Input text to classify.

        Returns:
            Tuple of (intent_label, confidence) where:
            - intent_label: "TRIP", "NOT_TRIP", or "UNKNOWN"
            - confidence: Float between 0.0 and 1.0
        """
        if len(text.strip()) < 3:
            return ("UNKNOWN", 0.5)

        entities = self.ner_model.predict_and_extract(text)
        has_dep = entities.get("departure") is not None
        has_dest = entities.get("destination") is not None

        if has_dep and has_dest:
            return ("TRIP", 0.9)
        if has_dep or has_dest:
            return ("TRIP", 0.7)
        return ("NOT_TRIP", 0.8)

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> List[Tuple[str, float]]:
        """Classify multiple texts via batched NER.

        Args:
            texts: List of input texts to classify.
            batch_size: Batch size for processing.
            progress_callback: Optional callback(current, total).

        Returns:
            List of (intent_label, confidence) tuples.
        """
        total = len(texts)
        all_results: List[Tuple[str, float]] = []

        entity_results = self.ner_model.predict_batch(texts, batch_size=batch_size)

        for text, entities in zip(texts, entity_results):
            if len(text.strip()) < 3:
                all_results.append(("UNKNOWN", 0.5))
                continue

            has_dep = entities.get("departure") is not None
            has_dest = entities.get("destination") is not None

            if has_dep and has_dest:
                all_results.append(("TRIP", 0.9))
            elif has_dep or has_dest:
                all_results.append(("TRIP", 0.7))
            else:
                all_results.append(("NOT_TRIP", 0.8))

        if progress_callback:
            progress_callback(total, total)

        return all_results
