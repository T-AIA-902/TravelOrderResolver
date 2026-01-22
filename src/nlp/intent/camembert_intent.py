"""
CamemBERT-based zero-shot intent classifier.

Uses the transformers zero-shot-classification pipeline to classify
travel intent without fine-tuning.
"""

from typing import Callable, List, Tuple

from ..interfaces import IntentClassifier


class CamembertIntentClassifier(IntentClassifier):
    """
    Zero-shot intent classifier using CamemBERT.

    Uses hypothesis templates to classify text as travel request or not.
    """

    def __init__(self, model_name: str = "almanach/camembert-base") -> None:
        """
        Initialize the CamemBERT intent classifier.

        Args:
            model_name: HuggingFace model name (default: almanach/camembert-base)
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers not available. Install with: pip install transformers")

        print(f"Loading CamemBERT for zero-shot classification: {model_name}...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
        )
        self.labels = ["demande de voyage en train", "autre question"]
        print("CamemBERT intent classifier ready")

    @property
    def name(self) -> str:
        return "CamemBERT"

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the input text using zero-shot classification.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (intent_label, confidence) where:
            - intent_label: "TRIP" or "NOT_TRIP"
            - confidence: Float between 0.0 and 1.0
        """
        # Handle empty or very short text
        if len(text.strip()) < 3:
            return ("UNKNOWN", 0.5)

        # Run zero-shot classification
        result = self.classifier(text, self.labels)

        # Determine intent based on top label
        is_trip = result["labels"][0] == "demande de voyage en train"
        confidence = result["scores"][0]

        return ("TRIP" if is_trip else "NOT_TRIP", confidence)

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 128,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> List[Tuple[str, float]]:
        """
        Classify multiple texts in batches for efficiency.

        Args:
            texts: List of input texts to classify
            batch_size: Number of texts per batch (default: 128)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of (intent_label, confidence) tuples
        """
        results: List[Tuple[str, float]] = []
        total = len(texts)

        # Process in batches
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]

            # Filter out empty/short texts and track their indices
            valid_texts = []
            valid_indices = []
            batch_results: List[Tuple[str, float]] = [("UNKNOWN", 0.5)] * len(batch)

            for j, text in enumerate(batch):
                if len(text.strip()) >= 3:
                    valid_texts.append(text)
                    valid_indices.append(j)

            # Run batched classification on valid texts
            if valid_texts:
                batch_outputs = self.classifier(valid_texts, self.labels)

                # Handle single result (not a list)
                if isinstance(batch_outputs, dict):
                    batch_outputs = [batch_outputs]

                for idx, output in zip(valid_indices, batch_outputs):
                    is_trip = output["labels"][0] == "demande de voyage en train"
                    confidence = output["scores"][0]
                    batch_results[idx] = ("TRIP" if is_trip else "NOT_TRIP", confidence)

            results.extend(batch_results)

            # Report progress after each batch
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return results
