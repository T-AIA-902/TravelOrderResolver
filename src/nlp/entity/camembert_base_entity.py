"""
CamemBERT base (zero-shot) entity extractor.

Uses the pre-trained Jean-Baptiste/camembert-ner model for generic NER
without any task-specific fine-tuning. Detects PER, LOC, ORG, MISC entities
and assigns roles by position (like SpaCy).

This serves as a baseline to compare against the fine-tuned CamemBERT NER
which uses custom BIO labels (B-DEP, B-DEST, B-STEP).
"""

from typing import Any, Callable, Dict, List, Optional

from ..interfaces import EntityExtractor


class CamembertBaseEntityExtractor(EntityExtractor):
    """
    Zero-shot CamemBERT NER using a pre-trained generic NER model.
    No task-specific fine-tuning — assigns roles by position only.
    """

    def __init__(self, device: str = "auto") -> None:
        from src.utils.device import get_torch_device

        self._device = get_torch_device(device)
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return

        from transformers import pipeline

        print("Loading CamemBERT base NER (Jean-Baptiste/camembert-ner)...")
        device_idx = 0 if self._device == "cuda" else -1
        self._pipeline = pipeline(
            "ner",
            model="Jean-Baptiste/camembert-ner",
            aggregation_strategy="simple",
            device=device_idx,
        )
        print(f"CamemBERT base NER ready (device: {self._device})")

    @property
    def name(self) -> str:
        return "CamemBERT (base)"

    def extract(self, text: str) -> Dict[str, Any]:
        if len(text.strip()) < 3:
            return {"departure": None, "destination": None, "intermediate": []}

        self._ensure_loaded()
        entities = self._pipeline(text)

        # Keep only location entities (LOC)
        locations = [
            ent["word"].strip()
            for ent in entities
            if ent["entity_group"] == "LOC" and ent["score"] > 0.5
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_locations = []
        for loc in locations:
            clean = loc.strip("▁ ")
            if clean and clean not in seen:
                seen.add(clean)
                unique_locations.append(clean)

        # Positional assignment (same as SpaCy — no task-specific knowledge)
        result: Dict[str, Any] = {
            "departure": None,
            "destination": None,
            "intermediate": [],
        }
        if len(unique_locations) == 0:
            pass
        elif len(unique_locations) == 1:
            result["destination"] = unique_locations[0]
        elif len(unique_locations) == 2:
            result["departure"] = unique_locations[0]
            result["destination"] = unique_locations[1]
        else:
            result["departure"] = unique_locations[0]
            result["destination"] = unique_locations[-1]
            result["intermediate"] = unique_locations[1:-1]

        return result

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(texts)
        for i, text in enumerate(texts):
            results.append(self.extract(text))
            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(i + 1, total)
        if progress_callback:
            progress_callback(total, total)
        return results
