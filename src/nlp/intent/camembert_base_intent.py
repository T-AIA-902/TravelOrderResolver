"""
CamemBERT base (zero-shot) intent classifier.

Uses the pre-trained CamemBERT model with zero-shot classification
(no fine-tuning). Classifies text as TRIP or NOT_TRIP using candidate labels.

This serves as a baseline to compare against the fine-tuned CamemBERT
which derives intent from NER results.
"""

from typing import Any, Callable, List, Optional, Tuple

from ..interfaces import IntentClassifier


class CamembertBaseIntentClassifier(IntentClassifier):
    """
    Zero-shot CamemBERT intent classifier using HuggingFace zero-shot pipeline.
    No task-specific fine-tuning.
    """

    def __init__(self, device: str = "auto") -> None:
        from src.utils.device import get_torch_device

        self._device = get_torch_device(device)
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return

        from transformers import pipeline

        print("Loading CamemBERT base zero-shot classifier...")
        device_idx = 0 if self._device == "cuda" else -1
        self._pipeline = pipeline(
            "zero-shot-classification",
            model="BaptisteDoworWorko/camembert-base-nli",
            device=device_idx,
        )
        print(f"CamemBERT base intent classifier ready (device: {self._device})")

    @property
    def name(self) -> str:
        return "CamemBERT (base)"

    def classify(self, text: str) -> Tuple[str, float]:
        if len(text.strip()) < 3:
            return ("UNKNOWN", 0.5)

        self._ensure_loaded()

        result = self._pipeline(
            text,
            candidate_labels=["demande de voyage en train", "autre sujet"],
        )

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        if "voyage" in top_label and top_score > 0.5:
            return ("TRIP", round(top_score, 2))
        else:
            return ("NOT_TRIP", round(top_score, 2))

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        total = len(texts)
        for i, text in enumerate(texts):
            results.append(self.classify(text))
            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(i + 1, total)
        if progress_callback:
            progress_callback(total, total)
        return results
