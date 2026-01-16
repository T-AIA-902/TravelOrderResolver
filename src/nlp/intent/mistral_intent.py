"""
Mistral intent classifier stub.

TODO: Implement LLM-based intent classification.
"""

from typing import Tuple

from ..interfaces import IntentClassifier


class MistralIntentClassifier(IntentClassifier):
    """
    Mistral intent classifier (not yet implemented).

    Will use prompted LLM generation for intent classification.
    """

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1") -> None:
        """
        Initialize the Mistral intent classifier.

        Args:
            model_name: HuggingFace model name
        """
        raise NotImplementedError(
            "MistralIntentClassifier is not yet implemented. "
            "Please use RegexIntentClassifier or CamembertIntentClassifier."
        )

    @property
    def name(self) -> str:
        return "Mistral"

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify intent using Mistral."""
        raise NotImplementedError("MistralIntentClassifier.classify() not implemented")
