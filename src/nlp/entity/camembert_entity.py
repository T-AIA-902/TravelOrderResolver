"""
CamemBERT fine-tuned NER entity extractor.

Uses CamembertForTokenClassification with BIO tags (B-DEP, I-DEP,
B-DEST, I-DEST, B-STEP, I-STEP) to extract departure, destination,
and intermediate stations from text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal

from ..interfaces import EntityExtractor

DeviceType = Literal["auto", "cuda", "cpu"]


class CamembertEntityExtractor(EntityExtractor):
    """Fine-tuned CamemBERT NER entity extractor.

    Uses token-level classification to extract travel entities
    instead of string matching.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: DeviceType = "auto",
        ner_model: Any | None = None,
    ) -> None:
        """Initialize the fine-tuned CamemBERT entity extractor.

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
        """Return the extractor name."""
        return "CamemBERT"

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract travel entities from the input text using NER.

        Args:
            text: Input text to process.

        Returns:
            Dictionary with keys:
            - departure: Optional[str] - Starting location
            - destination: Optional[str] - Ending location
            - intermediate: List[str] - Intermediate stops
        """
        result: Dict[str, Any] = self.ner_model.predict_and_extract(text)
        return result

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 128,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> List[Dict[str, Any]]:
        """Extract entities from multiple texts.

        Args:
            texts: List of input texts to process.
            batch_size: Batch size for processing.
            progress_callback: Optional callback(current, total).

        Returns:
            List of entity dictionaries.
        """
        results: List[Dict[str, Any]] = self.ner_model.predict_batch(texts, batch_size=batch_size)

        if progress_callback:
            progress_callback(len(texts), len(texts))

        return results
