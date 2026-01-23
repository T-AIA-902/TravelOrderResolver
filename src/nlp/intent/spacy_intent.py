"""
SpaCy-based intent classifier using NER for intent inference.

Uses spaCy's NER to detect location entities and infer travel intent.
"""

from typing import Callable, List, Literal, Optional, Tuple

from ..interfaces import IntentClassifier

DeviceType = Literal["auto", "cuda", "cpu"]


class SpacyIntentClassifier(IntentClassifier):
    """
    Classify intent using spaCy NER.

    Strategy: If text contains location entities (LOC, GPE),
    it's likely a travel request (TRIP).

    Note: This is a heuristic-based classifier and may have lower
    accuracy than regex or trained models.
    """

    def __init__(
        self,
        model_name: str = "fr_core_news_lg",
        device: DeviceType = "auto",
    ) -> None:
        """
        Initialize the SpaCy intent classifier.

        Args:
            model_name: Name of the spaCy model to load (default: fr_core_news_lg)
            device: Device to use - "auto", "cuda", or "cpu" (default: auto)
        """
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "spacy not available. Install with: pip install spacy && "
                "python -m spacy download fr_core_news_lg"
            )

        from src.utils.device import setup_spacy_device

        self.use_gpu = setup_spacy_device(device)
        print(f"Loading spaCy model for intent: {model_name}...")
        self.nlp = spacy.load(model_name)
        device_str = "GPU" if self.use_gpu else "CPU"
        print(f"SpaCy intent model loaded successfully (device: {device_str})")

    @property
    def name(self) -> str:
        return "SpaCy"

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify intent based on presence of location entities.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (intent_label, confidence) where:
            - intent_label: One of "TRIP", "NOT_TRIP"
            - confidence: Float between 0.0 and 1.0
        """
        doc = self.nlp(text)

        loc_entities = [ent for ent in doc.ents if ent.label_ in ["LOC", "GPE"]]

        if loc_entities:
            # More locations = higher confidence
            confidence = min(0.5 + len(loc_entities) * 0.1, 0.8)
            return ("TRIP", confidence)

        return ("NOT_TRIP", 0.5)

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 128,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Batch classification using nlp.pipe().

        Args:
            texts: List of input texts to classify
            batch_size: Number of texts per batch (default: 128)
            progress_callback: Optional callback(processed, total) for progress updates

        Returns:
            List of (intent_label, confidence) tuples
        """
        results: List[Tuple[str, float]] = []
        total = len(texts)

        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=batch_size)):
            if progress_callback and i % batch_size == 0:
                progress_callback(i, total)

            loc_entities = [ent for ent in doc.ents if ent.label_ in ["LOC", "GPE"]]

            if loc_entities:
                confidence = min(0.5 + len(loc_entities) * 0.1, 0.8)
                results.append(("TRIP", confidence))
            else:
                results.append(("NOT_TRIP", 0.5))

        if progress_callback:
            progress_callback(total, total)

        return results
