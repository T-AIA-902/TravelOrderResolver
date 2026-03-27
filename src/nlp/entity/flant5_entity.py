"""
Flan-T5 entity extractor.

Uses prompted seq2seq generation for extracting travel entities
(departure, destination, intermediate stops) from French text.
"""

import re
from typing import Any, Dict, List, Optional

from ..interfaces import EntityExtractor


class FlanT5EntityExtractor(EntityExtractor):
    """
    Flan-T5 entity extractor using prompt engineering.

    Uses structured prompts to extract departure, destination,
    and intermediate stations from travel requests.
    """

    # Prompt used during fine-tuning
    PROMPT_TEMPLATE = "Extrais les villes: {text}"

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the Flan-T5 entity extractor.

        Args:
            model_name: Model path or HuggingFace model name.
                       Defaults to local fine-tuned model.
        """
        from ..models.flan_t5_model import get_flan_t5_loader

        self.model_loader = get_flan_t5_loader(model_name)
        self._model_name = self.model_loader._model_name
        self.model_loader.load()
        print("Flan-T5 entity extractor ready")

    @property
    def name(self) -> str:
        """Return the model name."""
        return "Flan-T5"

    def _parse_output(self, output: str) -> Dict[str, Any]:
        """
        Parse model output to entity dictionary.

        Expected format: "DEPART: Paris | ARRIVEE: Lyon | VIA: aucun"

        Args:
            output: Raw model output

        Returns:
            Dictionary with departure, destination, intermediate
        """
        result: Dict[str, Any] = {
            "departure": None,
            "destination": None,
            "intermediate": [],
        }

        output_clean = output.strip()

        # Parse DEPART
        depart_match = re.search(
            r"DEPART\s*:\s*([^|]+?)(?:\s*\||$)",
            output_clean,
            re.IGNORECASE,
        )
        if depart_match:
            value = depart_match.group(1).strip()
            if value.lower() not in ["aucun", "none", "n/a", ""]:
                result["departure"] = value

        # Parse ARRIVEE
        arrivee_match = re.search(
            r"ARRIVEE\s*:\s*([^|]+?)(?:\s*\||$)",
            output_clean,
            re.IGNORECASE,
        )
        if arrivee_match:
            value = arrivee_match.group(1).strip()
            if value.lower() not in ["aucun", "none", "n/a", ""]:
                result["destination"] = value

        # Parse VIA (intermediate stops)
        via_match = re.search(
            r"VIA\s*:\s*([^|]+?)(?:\s*\||$)",
            output_clean,
            re.IGNORECASE,
        )
        if via_match:
            value = via_match.group(1).strip()
            if value.lower() not in ["aucun", "none", "n/a", ""]:
                intermediates = [
                    v.strip()
                    for v in value.split(",")
                    if v.strip() and v.strip().lower() != "aucun"
                ]
                result["intermediate"] = intermediates

        return result

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract travel entities from the input text.

        Args:
            text: Input text to process

        Returns:
            Dictionary with keys:
            - departure: Optional[str] - Starting location
            - destination: Optional[str] - Ending location
            - intermediate: List[str] - Intermediate stops
        """
        if not text or len(text.strip()) < 3:
            return {
                "departure": None,
                "destination": None,
                "intermediate": [],
            }

        prompt = self.PROMPT_TEMPLATE.format(text=text)

        output = self.model_loader.generate(
            prompt,
            max_new_tokens=128,
            temperature=0.1,
        )

        return self._parse_output(output)

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        progress_callback: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size (for future optimization)
            progress_callback: Optional callback for progress reporting

        Returns:
            List of entity dictionaries
        """
        results = []
        for i, text in enumerate(texts):
            results.append(self.extract(text))
            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(i + 1, len(texts))
        return results
