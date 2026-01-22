"""
SpaCy-based entity extractor.

Uses spaCy NER to identify location entities in travel text.
"""

from typing import Any, Callable, Dict, List, Optional

from ..interfaces import EntityExtractor


class SpacyEntityExtractor(EntityExtractor):
    """
    Extract travel entities using spaCy NER.

    Uses spaCy's French model to identify location entities (LOC, GPE),
    then applies heuristics to classify them as departure, destination,
    or intermediate stops.
    """

    def __init__(self, model_name: str = "fr_core_news_lg") -> None:
        """
        Initialize the entity extractor with a spaCy model.

        Args:
            model_name: Name of the spaCy model to load (default: fr_core_news_lg)
        """
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "spacy not available. Install with: pip install spacy && "
                "python -m spacy download fr_core_news_lg"
            )

        print(f"Loading spaCy model: {model_name}...")
        self.nlp = spacy.load(model_name)
        print("Model loaded successfully!")

    @property
    def name(self) -> str:
        return "SpaCy"

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract travel entities from the input text.

        Uses spaCy NER to find location entities, then applies heuristics:
        - First location = departure
        - Last location = destination
        - Middle locations (with intermediate keywords) = intermediate

        Args:
            text: Input text to process

        Returns:
            Dictionary with keys:
            - departure: Optional[str] - Starting location
            - destination: Optional[str] - Ending location
            - intermediate: List[str] - Intermediate stops
        """
        # Run spaCy NER pipeline
        doc = self.nlp(text)

        # Extract all location entities with positions
        location_entities: List[Dict[str, Any]] = [
            {"text": ent.text, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
            if ent.label_ in ["LOC", "GPE"]
        ]

        # Identify intermediate keywords
        intermediate_keywords = ["via", "par", "passant par", "en passant par", "arret"]
        text_lower = text.lower()

        # Find locations that appear after intermediate keywords
        intermediate_locs: set[str] = set()
        for keyword in intermediate_keywords:
            if keyword in text_lower:
                keyword_pos = text_lower.find(keyword)
                for loc in location_entities:
                    # Only consider locations within ~30 chars after keyword
                    if keyword_pos < loc["start"] < keyword_pos + 30:
                        intermediate_locs.add(loc["text"])

        # Initialize result
        result: Dict[str, Any] = {
            "departure": None,
            "destination": None,
            "intermediate": [],
        }

        if len(location_entities) == 0:
            return result
        elif len(location_entities) == 1:
            result["destination"] = location_entities[0]["text"]
        elif len(location_entities) == 2:
            result["departure"] = location_entities[0]["text"]
            result["destination"] = location_entities[1]["text"]
        else:
            locations = [loc["text"] for loc in location_entities]

            if intermediate_locs:
                result["departure"] = locations[0]
                result["destination"] = locations[-1]
                result["intermediate"] = [
                    loc for loc in locations[1:-1] if loc in intermediate_locs
                ]
            else:
                result["departure"] = locations[0]
                result["destination"] = locations[-1]
                result["intermediate"] = locations[1:-1]

        return result

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

        # Process with nlp.pipe for efficiency
        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=batch_size)):
            # Report progress every batch_size items
            if progress_callback and i % batch_size == 0:
                progress_callback(i, total)
            # Extract location entities
            location_entities: List[Dict[str, Any]] = [
                {"text": ent.text, "start": ent.start_char, "end": ent.end_char}
                for ent in doc.ents
                if ent.label_ in ["LOC", "GPE"]
            ]

            # Identify intermediate keywords
            text_lower = doc.text.lower()
            intermediate_keywords = ["via", "par", "passant par", "en passant par", "arret"]

            intermediate_locs: set[str] = set()
            for keyword in intermediate_keywords:
                if keyword in text_lower:
                    keyword_pos = text_lower.find(keyword)
                    for loc in location_entities:
                        if keyword_pos < loc["start"] < keyword_pos + 30:
                            intermediate_locs.add(loc["text"])

            # Build result
            result: Dict[str, Any] = {
                "departure": None,
                "destination": None,
                "intermediate": [],
            }

            if len(location_entities) == 0:
                pass
            elif len(location_entities) == 1:
                result["destination"] = location_entities[0]["text"]
            elif len(location_entities) == 2:
                result["departure"] = location_entities[0]["text"]
                result["destination"] = location_entities[1]["text"]
            else:
                locations = [loc["text"] for loc in location_entities]
                if intermediate_locs:
                    result["departure"] = locations[0]
                    result["destination"] = locations[-1]
                    result["intermediate"] = [
                        loc for loc in locations[1:-1] if loc in intermediate_locs
                    ]
                else:
                    result["departure"] = locations[0]
                    result["destination"] = locations[-1]
                    result["intermediate"] = locations[1:-1]

            results.append(result)

        # Final progress callback
        if progress_callback:
            progress_callback(total, total)

        return results
