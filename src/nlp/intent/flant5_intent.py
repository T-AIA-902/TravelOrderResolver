"""
Flan-T5 intent classifier stub.

TODO: Implement seq2seq-based intent classification.
"""

from typing import Tuple

from ..interfaces import IntentClassifier


class FlanT5IntentClassifier(IntentClassifier):
    """
    Flan-T5 intent classifier (not yet implemented).

    Will use prompted seq2seq generation for intent classification.
    """

    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        """
        Initialize the Flan-T5 intent classifier.

        Args:
            model_name: HuggingFace model name
        """
        raise NotImplementedError(
            "FlanT5IntentClassifier is not yet implemented. "
            "Please use RegexIntentClassifier or CamembertIntentClassifier."
        )

    @property
    def name(self) -> str:
        return "Flan-T5"

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify intent using Flan-T5."""
        raise NotImplementedError("FlanT5IntentClassifier.classify() not implemented")
