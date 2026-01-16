"""
Mistral entity extractor stub.

TODO: Implement LLM-based entity extraction.
"""

from typing import Any, Dict

from ..interfaces import EntityExtractor


class MistralEntityExtractor(EntityExtractor):
    """
    Mistral entity extractor (not yet implemented).

    Will use prompted LLM generation for entity extraction.
    """

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1") -> None:
        """
        Initialize the Mistral entity extractor.

        Args:
            model_name: HuggingFace model name
        """
        raise NotImplementedError(
            "MistralEntityExtractor is not yet implemented. "
            "Please use RegexEntityExtractor, SpacyEntityExtractor, "
            "or CamembertEntityExtractor."
        )

    @property
    def name(self) -> str:
        return "Mistral"

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract entities using Mistral."""
        raise NotImplementedError("MistralEntityExtractor.extract() not implemented")
