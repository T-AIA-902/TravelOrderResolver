"""
SpaCy-based entity extractor.

Uses spaCy NER to identify location entities in travel text.
SpaCy's fr_core_news_lg model detects LOC/GPE entities but has no notion
of departure, destination, or intermediate stops — role assignment relies
on positional heuristics (first = departure, last = destination).
This is a known limitation compared to fine-tuned models (CamemBERT, Flan-T5).
"""

from typing import Any, Callable, Dict, List, Literal, Optional

from ..interfaces import EntityExtractor

DeviceType = Literal["auto", "cuda", "cpu"]


class SpacyEntityExtractor(EntityExtractor):
    """
    Extract travel entities using spaCy NER.

    Uses spaCy's French model to identify location entities (LOC, GPE),
    then assigns roles based on position only:
    - First location = departure
    - Last location = destination
    - Everything in between = intermediate stops

    This approach has no understanding of linguistic cues like "via" or
    "en passant par", which is a limitation that fine-tuned models address.
    """

    def __init__(
        self,
        model_name: str = "fr_core_news_lg",
        device: DeviceType = "auto",
    ) -> None:
        """
        Initialize the entity extractor with a spaCy model.

        Args:
            model_name: Name of the spaCy model to load (default: fr_core_news_lg)
            device: Device to use - "auto", "cuda", or "cpu" (default: auto)
                   Note: Requires spacy[cuda] for GPU support
        """
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "spacy not available. Install with: pip install spacy && "
                "python -m spacy download fr_core_news_lg"
            )

        from src.utils.device import setup_spacy_device

        # Setup GPU before loading model (must be called before spacy.load)
        self.use_gpu = setup_spacy_device(device)

        print(f"Loading spaCy model: {model_name}...")
        self.nlp = spacy.load(model_name)
        device_str = "GPU" if self.use_gpu else "CPU"
        print(f"SpaCy model loaded successfully (device: {device_str})")

    @property
    def name(self) -> str:
        return "SpaCy"

    @staticmethod
    def _assign_roles(locations: List[str]) -> Dict[str, Any]:
        """Assign departure/destination/intermediate based on position only."""
        result: Dict[str, Any] = {
            "departure": None,
            "destination": None,
            "intermediate": [],
        }
        if len(locations) == 0:
            pass
        elif len(locations) == 1:
            result["destination"] = locations[0]
        elif len(locations) == 2:
            result["departure"] = locations[0]
            result["destination"] = locations[1]
        else:
            result["departure"] = locations[0]
            result["destination"] = locations[-1]
            result["intermediate"] = locations[1:-1]
        return result

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract travel entities from the input text.

        Uses spaCy NER to find location entities, then assigns roles
        purely by position (first = departure, last = destination).

        Args:
            text: Input text to process

        Returns:
            Dictionary with keys:
            - departure: Optional[str] - Starting location
            - destination: Optional[str] - Ending location
            - intermediate: List[str] - Intermediate stops
        """
        doc = self.nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE"]]
        return self._assign_roles(locations)

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 128,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple texts using batched processing.

        Uses spaCy's nlp.pipe() for efficient batch processing.

        Args:
            texts: List of input texts to process
            batch_size: Number of texts per batch (default: 128)
            progress_callback: Optional callback(processed, total) for progress updates

        Returns:
            List of entity dictionaries
        """
        results: List[Dict[str, Any]] = []
        total = len(texts)

        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=batch_size)):
            if progress_callback and i % batch_size == 0:
                progress_callback(i, total)

            locations = [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE"]]
            results.append(self._assign_roles(locations))

        if progress_callback:
            progress_callback(total, total)

        return results
