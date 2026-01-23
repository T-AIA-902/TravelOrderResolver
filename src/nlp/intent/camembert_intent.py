"""
CamemBERT-based zero-shot intent classifier.

Uses the transformers zero-shot-classification pipeline to classify
travel intent without fine-tuning.
"""

from typing import Callable, List, Literal, Tuple

from ..interfaces import IntentClassifier

DeviceType = Literal["auto", "cuda", "cpu"]


class CamembertIntentClassifier(IntentClassifier):
    """
    Zero-shot intent classifier using CamemBERT.

    Uses hypothesis templates to classify text as travel request or not.
    """

    def __init__(
        self,
        model_name: str = "almanach/camembert-base",
        device: DeviceType = "auto",
    ) -> None:
        """
        Initialize the CamemBERT intent classifier.

        Args:
            model_name: HuggingFace model name (default: almanach/camembert-base)
            device: Device to use - "auto", "cuda", or "cpu" (default: auto)
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers not available. Install with: pip install transformers")

        from src.utils.device import get_torch_device

        self.device = get_torch_device(device)

        print(f"Loading CamemBERT for zero-shot classification: {model_name}...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device,
        )
        self.labels = ["demande de voyage en train", "autre question"]
        print(f"CamemBERT intent classifier ready (device: {self.device})")

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
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> List[Tuple[str, float]]:
        """
        Classify multiple texts efficiently using HuggingFace Dataset.

        Uses Dataset-based batching for optimal GPU throughput.

        Args:
            texts: List of input texts to classify
            batch_size: Batch size for GPU processing (default: 32)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of (intent_label, confidence) tuples
        """
        from src.nlp.utils.hf_batching import run_pipeline_batched

        total = len(texts)

        # Filter invalid texts, track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if len(text.strip()) >= 3:
                valid_texts.append(text)
                valid_indices.append(i)

        # Initialize all results as UNKNOWN
        all_results: List[Tuple[str, float]] = [("UNKNOWN", 0.5)] * total

        if valid_texts:
            # Single batched call with Dataset optimization
            outputs = run_pipeline_batched(
                self.classifier,
                valid_texts,
                batch_size=batch_size,
                candidate_labels=self.labels,
            )

            # Map results back to original indices
            for idx, output in zip(valid_indices, outputs):
                is_trip = output["labels"][0] == "demande de voyage en train"
                confidence = output["scores"][0]
                all_results[idx] = ("TRIP" if is_trip else "NOT_TRIP", confidence)

        if progress_callback:
            progress_callback(total, total)

        return all_results
