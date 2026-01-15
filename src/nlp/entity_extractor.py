"""
Entity extraction module using spaCy NER.

This module extracts travel-related entities (departure, destination, intermediate stops)
from French sentences using spaCy's pre-trained transformer model.
"""

import spacy
from typing import Dict, List, Optional
from pathlib import Path

try:
    from src.nlp.fuzzy_matcher import StationMatcher
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False


class SpacyEntityExtractor:
    """
    Extract travel entities (departure, destination, intermediate) using spaCy NER.

    This class uses spaCy's French transformer model to identify location entities
    in travel-related sentences, then applies heuristics to classify them as
    departure, destination, or intermediate stops.
    """

    def __init__(self, model_name: str = "fr_core_news_lg"):
        """
        Initialize the entity extractor with a spaCy model.

        Args:
            model_name: Name of the spaCy model to load (default: fr_core_news_lg)
        """
        print(f"Loading spaCy model: {model_name}...")
        self.nlp = spacy.load(model_name)
        print("Model loaded successfully!")

    def extract_entities(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract departure, destination, and intermediate stops from text.

        **How it works:**
        1. Run spaCy NER to find all location entities (LOC, GPE)
        2. Detect intermediate keywords ("via", "par", "en passant par")
        3. Apply heuristics:
           - First location = departure
           - Last location = destination
           - Middle locations with intermediate keywords = intermediate stops

        **Known limitations:**
        - May misclassify intermediate stops if word order is ambiguous
        - Example: "De Paris à Lyon en passant par Dijon" may detect Lyon
          as intermediate instead of Dijon, depending on entity detection order
        - Does not handle misspellings (e.g., "Pari" instead of "Paris")

        Args:
            text: Input sentence (e.g., "Je veux aller de Paris à Lyon")

        Returns:
            Dictionary with extracted entities:
            {
                "departure": "Paris",
                "destination": "Lyon",
                "intermediate": []
            }

        Example:
            >>> extractor = SpacyEntityExtractor()
            >>> result = extractor.extract_entities("De Paris à Lyon via Dijon")
            >>> print(result)
            {'departure': 'Paris', 'destination': 'Dijon', 'intermediate': ['Lyon']}
        """
        # Step 1: Run spaCy NER pipeline on the text
        doc = self.nlp(text)

        # Step 2: Extract all location entities WITH their positions
        # spaCy identifies entities with labels:
        # - LOC = Location (geographic places)
        # - GPE = Geopolitical Entity (cities, countries)
        location_entities = [
            {"text": ent.text, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents if ent.label_ in ["LOC", "GPE"]
        ]

        # Step 3: Identify intermediate keywords
        # Keywords that indicate intermediate stops
        intermediate_keywords = ["via", "par", "passant par", "en passant par", "arret"]
        text_lower = text.lower()

        # Find locations that appear after intermediate keywords
        intermediate_locs = set()  # Use set to avoid duplicates
        for keyword in intermediate_keywords:
            if keyword in text_lower:
                keyword_pos = text_lower.find(keyword)
                # Find locations that come IMMEDIATELY after this keyword (not all after)
                for loc in location_entities:
                    # Only consider locations within ~30 characters after the keyword
                    if keyword_pos < loc["start"] < keyword_pos + 30:
                        intermediate_locs.add(loc["text"])

        # Step 4: Apply heuristics to classify locations
        result = {
            "departure": None,
            "destination": None,
            "intermediate": []
        }

        if len(location_entities) == 0:
            # No locations found
            return result
        elif len(location_entities) == 1:
            # Only one location: assume it's the destination
            result["destination"] = location_entities[0]["text"]
        elif len(location_entities) == 2:
            # Two locations: first = departure, second = destination
            result["departure"] = location_entities[0]["text"]
            result["destination"] = location_entities[1]["text"]
        else:
            # More than 2 locations: use intermediate keywords logic
            locations = [loc["text"] for loc in location_entities]

            if intermediate_locs:
                # We found intermediate keywords
                # First location = departure
                # Locations marked by intermediate keywords = intermediate
                # Last location in sentence = destination (even if it appears before intermediate)
                result["departure"] = locations[0]
                result["destination"] = locations[-1]  # Last location in order of appearance

                # Intermediate = locations marked by keywords, excluding departure and destination
                result["intermediate"] = [
                    loc for loc in locations[1:-1] if loc in intermediate_locs
                ]
            else:
                # No intermediate keywords: use simple order
                result["departure"] = locations[0]
                result["destination"] = locations[-1]
                result["intermediate"] = locations[1:-1]

        return result

    def extract_entities_with_details(self, text: str) -> Dict:
        """
        Extract entities with additional details (positions, confidence, labels).

        This is useful for debugging and understanding what spaCy detects.

        Args:
            text: Input sentence

        Returns:
            Dictionary with detailed entity information

        Example:
            >>> extractor = SpacyEntityExtractor()
            >>> result = extractor.extract_entities_with_details("De Paris à Lyon")
            >>> print(result)
            {
                'text': 'De Paris à Lyon',
                'entities': [
                    {'text': 'Paris', 'label': 'LOC', 'start': 3, 'end': 8},
                    {'text': 'Lyon', 'label': 'LOC', 'start': 11, 'end': 15}
                ],
                'departure': 'Paris',
                'destination': 'Lyon',
                'intermediate': []
            }
        """
        # Run spaCy NER
        doc = self.nlp(text)

        # Collect all location entities with details
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["LOC", "GPE"]:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

        # Extract departure/destination using the simple method
        extracted = self.extract_entities(text)

        return {
            "text": text,
            "entities": entities,
            "departure": extracted["departure"],
            "destination": extracted["destination"],
            "intermediate": extracted["intermediate"]
        }


class FuzzyEntityExtractor(SpacyEntityExtractor):
    """
    Enhanced entity extractor with fuzzy matching to SNCF station database.

    Extends SpacyEntityExtractor by adding fuzzy matching to normalize extracted
    location names to official SNCF station names. This handles:
    - Misspellings (e.g., "pari" -> "Paris-Bercy")
    - Case variations (e.g., "MARSEILLE" -> "Marseille-St-Charles")
    - Missing characters (e.g., "Nante" -> "Nantes")
    """

    def __init__(
        self,
        model_name: str = "fr_core_news_lg",
        stations_path: Optional[str] = None,
        fuzzy_threshold: int = 75,
    ):
        """
        Initialize the fuzzy entity extractor.

        Args:
            model_name: spaCy model name (default: fr_core_news_lg)
            stations_path: Path to stations.json file (default: auto-detect)
            fuzzy_threshold: Minimum similarity score for fuzzy matching (default: 75)
        """
        # Initialize parent spaCy extractor
        super().__init__(model_name)

        # Initialize fuzzy matcher
        if not FUZZY_MATCHING_AVAILABLE:
            raise ImportError(
                "Fuzzy matching dependencies not available. "
                "Install with: pip install rapidfuzz unidecode"
            )

        self.fuzzy_matcher = StationMatcher(
            stations_path=stations_path, threshold=fuzzy_threshold
        )
        print(f"Fuzzy matching enabled (threshold: {fuzzy_threshold}%)")

    def extract_entities(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract and normalize entities using fuzzy matching.

        First runs spaCy NER to extract location entities, then normalizes
        each location to the closest SNCF station using fuzzy matching.

        Args:
            text: Input sentence

        Returns:
            Dictionary with normalized entities:
            {
                "departure": "Paris-Gare-de-Lyon",
                "destination": "Lyon-Perrache",
                "intermediate": []
            }

        Example:
            >>> extractor = FuzzyEntityExtractor()
            >>> result = extractor.extract_entities("De pari a lion")
            >>> print(result)
            {'departure': 'Paris-Bercy', 'destination': 'Lyon-St-Paul', 'intermediate': []}
        """
        # Step 1: Extract entities using spaCy (parent method)
        entities = super().extract_entities(text)

        # Step 2: Normalize each entity with fuzzy matching
        if entities["departure"]:
            match = self.fuzzy_matcher.match_station(entities["departure"])
            if match:
                entities["departure"] = match[0]  # Use matched station name
            # If no match, keep original spaCy entity

        if entities["destination"]:
            match = self.fuzzy_matcher.match_station(entities["destination"])
            if match:
                entities["destination"] = match[0]

        # Normalize intermediate stops
        normalized_intermediate = []
        for stop in entities["intermediate"]:
            match = self.fuzzy_matcher.match_station(stop)
            if match:
                normalized_intermediate.append(match[0])
            else:
                normalized_intermediate.append(stop)  # Keep original if no match
        entities["intermediate"] = normalized_intermediate

        return entities


def main():
    """
    Demo function to test the entity extractor on example sentences.
    """
    print("=" * 60)
    print("spaCy Entity Extractor - Demo")
    print("=" * 60)

    # Create extractor
    extractor = SpacyEntityExtractor()

    # Test sentences
    test_sentences = [
        "Je veux aller de Paris à Lyon",
        "De Marseille à Toulouse s'il vous plaît",
        "Paris Lyon",
        "De Paris à Lyon en passant par Dijon",
        "Quel temps fait-il ?",
    ]

    print("\nTesting entity extraction:\n")

    for sentence in test_sentences:
        result = extractor.extract_entities(sentence)
        print(f"Input:    {sentence}")
        print(f"Departure:    {result['departure']}")
        print(f"Destination:  {result['destination']}")
        print(f"Intermediate: {result['intermediate']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
