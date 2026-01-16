"""
Flan-T5 entity extractor stub.

TODO: Implement seq2seq-based entity extraction.
"""

from typing import Any, Dict

from ..interfaces import EntityExtractor


class FlanT5EntityExtractor(EntityExtractor):
    """
    Flan-T5 entity extractor (not yet implemented).

    Will use prompted seq2seq generation for entity extraction.
    """

    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        """
        Initialize the Flan-T5 entity extractor.

        Args:
            model_name: HuggingFace model name
        """
        raise NotImplementedError(
            "FlanT5EntityExtractor is not yet implemented. "
            "Please use RegexEntityExtractor, SpacyEntityExtractor, "
            "or CamembertEntityExtractor."
        )

    @property
    def name(self) -> str:
        return "Flan-T5"

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract entities using Flan-T5."""
        raise NotImplementedError("FlanT5EntityExtractor.extract() not implemented")
