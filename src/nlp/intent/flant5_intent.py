"""
Flan-T5 intent classifier.

Uses prompted seq2seq generation for intent classification.
Classifies French travel requests as TRIP, NOT_TRIP, NOT_FRENCH, or UNKNOWN.
"""

from typing import List, Tuple

from ..interfaces import IntentClassifier


class FlanT5IntentClassifier(IntentClassifier):
    """
    Flan-T5 intent classifier using prompt engineering.

    Uses French prompts to classify text into travel intent categories.
    """

    PROMPT_TEMPLATE = """Classifie cette phrase en francais.
Reponds uniquement par: VOYAGE, PAS_VOYAGE, PAS_FRANCAIS, ou INCONNU.

Phrase: "{text}"

Classification:"""

    OUTPUT_MAP = {
        "voyage": "TRIP",
        "pas_voyage": "NOT_TRIP",
        "pas_francais": "NOT_FRENCH",
        "inconnu": "UNKNOWN",
        "pas voyage": "NOT_TRIP",
        "pas francais": "NOT_FRENCH",
    }

    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        """
        Initialize the Flan-T5 intent classifier.

        Note: Uses the base model (not fine-tuned) because intent classification
        requires different training than entity extraction. The fine-tuned model
        is optimized for extracting cities, not classifying intents.

        Args:
            model_name: HuggingFace model name (e.g., "google/flan-t5-base")
        """
        from ..models.flan_t5_model import get_flan_t5_loader

        # Always use base model for intent - our fine-tuned model is for entities only
        self._model_name = model_name
        self.model_loader = get_flan_t5_loader(model_name)
        self.model_loader.load()
        print("Flan-T5 intent classifier ready")

    @property
    def name(self) -> str:
        """Return the model name."""
        return "Flan-T5"

    def _parse_output(self, output: str) -> Tuple[str, float]:
        """
        Parse model output to intent label.

        Args:
            output: Raw model output

        Returns:
            Tuple of (intent_label, confidence)
        """
        output_clean = output.strip().lower().replace("_", " ")

        for key, label in self.OUTPUT_MAP.items():
            if key in output_clean:
                return (label, 0.9)

        return ("UNKNOWN", 0.5)

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the input text using Flan-T5.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (intent_label, confidence)
            - intent_label: "TRIP", "NOT_TRIP", "NOT_FRENCH", or "UNKNOWN"
            - confidence: float between 0 and 1
        """
        if len(text.strip()) < 3:
            return ("UNKNOWN", 0.5)

        prompt = self.PROMPT_TEMPLATE.format(text=text)

        output = self.model_loader.generate(
            prompt,
            max_new_tokens=16,
            temperature=0.1,
        )

        return self._parse_output(output)

    def classify_batch(self, texts: List[str], batch_size: int = 16) -> List[Tuple[str, float]]:
        """
        Classify multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size (for future optimization)

        Returns:
            List of (intent_label, confidence) tuples
        """
        return [self.classify(text) for text in texts]
