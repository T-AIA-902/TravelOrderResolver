"""
Entity extraction module for travel order resolution.

This module provides multiple entity extraction backends:
- SpacyEntityExtractor: Uses spaCy fr_core_news_lg NER
- FuzzyEntityExtractor: SpaCy + RapidFuzz station matching
- CamembertEntityExtractor: Fine-tuned CamemBERT NER model

All extractors share a common interface: extract_entities(text) -> Dict
"""

from typing import Any, Dict, List, Optional, Union, cast

import spacy

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

    def extract_entities(self, text: str) -> Dict[str, Union[Optional[str], List[str]]]:
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
        location_entities: List[Dict[str, Any]] = [
            {"text": ent.text, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
            if ent.label_ in ["LOC", "GPE"]
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
        result: Dict[str, Union[Optional[str], List[str]]] = {
            "departure": None,
            "destination": None,
            "intermediate": [],
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
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )

        # Extract departure/destination using the simple method
        extracted = self.extract_entities(text)

        return {
            "text": text,
            "entities": entities,
            "departure": extracted["departure"],
            "destination": extracted["destination"],
            "intermediate": extracted["intermediate"],
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
        fuzzy_threshold: int = 75,
    ):
        """
        Initialize the fuzzy entity extractor.

        Args:
            model_name: spaCy model name (default: fr_core_news_lg)
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

        self.fuzzy_matcher = StationMatcher(threshold=fuzzy_threshold)
        print(f"Fuzzy matching enabled (threshold: {fuzzy_threshold}%)")

    def extract_entities(self, text: str) -> Dict[str, Union[Optional[str], List[str]]]:
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
        departure = cast(Optional[str], entities["departure"])
        if departure:
            match = self.fuzzy_matcher.match_station(departure)
            if match:
                entities["departure"] = match[0]  # Use matched station name
            # If no match, keep original spaCy entity

        destination = cast(Optional[str], entities["destination"])
        if destination:
            match = self.fuzzy_matcher.match_station(destination)
            if match:
                entities["destination"] = match[0]

        # Normalize intermediate stops
        intermediate = cast(List[str], entities["intermediate"])
        normalized_intermediate: List[str] = []
        for stop in intermediate:
            match = self.fuzzy_matcher.match_station(stop)
            if match:
                normalized_intermediate.append(match[0])
            else:
                normalized_intermediate.append(stop)  # Keep original if no match
        entities["intermediate"] = normalized_intermediate

        return entities


class CamembertEntityExtractor:
    """
    Extract entities using fine-tuned CamemBERT NER model.

    This extractor uses a CamemBERT model fine-tuned on French travel requests
    with BIO-tagged labels: B-DEP, I-DEP, B-DEST, I-DEST, O.

    Advantages over SpaCy:
    - Trained specifically on travel domain data
    - Better handling of French station names
    - Direct departure/destination classification (no heuristics needed)
    """

    def __init__(self, model_path: str = "models/camembert-ner") -> None:
        """
        Initialize the CamemBERT entity extractor.

        Args:
            model_path: Path to the fine-tuned CamemBERT model directory
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers not available. Install with: pip install transformers torch"
            )

        print(f"Loading CamemBERT NER model from {model_path}...")
        self.nlp = pipeline(  # type: ignore[call-overload]
            "ner", model=model_path, aggregation_strategy="simple"
        )
        print("CamemBERT model loaded successfully!")

    def extract_entities(self, text: str) -> Dict[str, Union[Optional[str], List[str]]]:
        """
        Extract departure and destination using CamemBERT NER.

        The model directly predicts DEP (departure) and DEST (destination) labels,
        so no heuristics are needed to classify entities.

        Args:
            text: Input sentence (e.g., "Je veux aller de Paris à Lyon")

        Returns:
            Dictionary with extracted entities:
            {
                "departure": "Paris",
                "destination": "Lyon",
                "intermediate": []
            }
        """
        results = self.nlp(text)

        departure: Optional[str] = None
        destination: Optional[str] = None

        for entity in results:
            label = entity["entity_group"]
            word = entity["word"].strip()

            if "DEP" in label and departure is None:
                departure = word
            elif "DEST" in label and destination is None:
                destination = word

        return {
            "departure": departure,
            "destination": destination,
            "intermediate": [],  # CamemBERT model doesn't handle intermediate stops yet
        }


class CamembertZeroShotExtractor:
    """
    Zero-shot entity extraction using camembert-base embeddings.

    This extractor uses the base CamemBERT model (no fine-tuning) to extract
    travel entities by comparing token embeddings with pre-computed station
    name embeddings using cosine similarity.

    This establishes a baseline for comparison with fine-tuned models.
    """

    def __init__(
        self,
        model_name: str = "almanach/camembert-base",
        threshold: float = 0.85,
    ) -> None:
        """
        Initialize the zero-shot CamemBERT extractor.

        Args:
            model_name: HuggingFace model name (default: almanach/camembert-base)
            threshold: Minimum cosine similarity for station matching (default: 0.85)
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers/torch not available. " "Install with: pip install transformers torch"
            )

        print(f"Loading CamemBERT base model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        self.threshold = threshold
        self.torch = torch

        # Load station database and precompute embeddings
        from src.data import StationDatabase

        self.station_db = StationDatabase()
        self.station_db.load()
        self._precompute_station_embeddings()
        print(f"CamemBERT zero-shot extractor ready (threshold: {threshold})")

    def _get_embedding(self, text: str) -> Any:
        """Get the mean pooled embedding for a text."""
        with self.torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=32
            )
            outputs = self.model(**inputs)
            # Mean pooling over token embeddings (excluding special tokens)
            attention_mask = inputs["attention_mask"]
            embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            sum_embeddings = (embeddings * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

    def _precompute_station_embeddings(self) -> None:
        """Precompute embeddings for all station names."""
        print("Precomputing station embeddings...")
        self.station_embeddings: Dict[str, Any] = {}
        self.city_names: List[str] = []

        # Get unique city names from station database
        cities = set()
        for station in self.station_db.get_all_stations():
            # Station is a dataclass with name and commune attributes
            city = station.commune if station.commune else station.name
            if city and len(city) >= 2:
                # Capitalize properly (commune is often uppercase)
                cities.add(city.title())

        self.city_names = sorted(cities)

        # Batch compute embeddings for efficiency
        batch_size = 64
        for i in range(0, len(self.city_names), batch_size):
            batch = self.city_names[i : i + batch_size]
            for city in batch:
                self.station_embeddings[city.lower()] = self._get_embedding(city)

        print(f"Computed embeddings for {len(self.station_embeddings)} cities")

    def _find_station_matches(self, text: str) -> List[Dict[str, Any]]:
        """
        Find potential station matches in text using embedding similarity.

        Uses a multi-stage approach:
        1. First check for exact multi-word/hyphenated station name matches
        2. Then check for single-word exact matches
        3. Finally use embedding similarity as fallback

        Returns list of matches with position, text, and similarity score.
        """
        matches = []
        text_lower = text.lower()

        # Common French words to exclude
        stop_words = {
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "de",
            "du",
            "au",
            "aux",
            "et",
            "ou",
            "mais",
            "donc",
            "car",
            "ni",
            "que",
            "qui",
            "quoi",
            "ce",
            "cette",
            "ces",
            "mon",
            "ma",
            "mes",
            "ton",
            "ta",
            "tes",
            "son",
            "sa",
            "ses",
            "notre",
            "votre",
            "leur",
            "leurs",
            "aller",
            "veux",
            "vouloir",
            "voudrais",
            "partir",
            "prendre",
            "pour",
            "par",
            "avec",
            "sans",
            "sur",
            "sous",
            "dans",
            "en",
            "svp",
            "stp",
            "merci",
            "bonjour",
            "train",
            "billet",
            "voyage",
            "passant",
            "via",
            "depuis",
            "vers",
            "jusque",
            "jusqu",
            "quel",
            "quelle",
            "comment",
            "temps",
            "fait",
            "demain",
            "aujourd",
            "go",
            "côté",
            "réunion",
            "place",
            "liaison",
            "entre",
            "partant",
            "provenance",
            "simple",
            "retour",
            "aller-retour",
            "aller-simple",
            "heure",
            "part",
            "arrive",
            "arrivée",
            "départ",
            "tgv",
            "ter",
        }

        # Stage 1: Check for exact city name matches in text
        # Sort cities by length (longest first) to match multi-word names first
        matched_positions: set[int] = set()
        cities_by_length = sorted(self.station_embeddings.keys(), key=len, reverse=True)

        # Word boundary characters
        boundary_chars = set(" .,;:!?()[]{}\"'\t\n-")

        for city in cities_by_length:
            city_lower = city.lower()

            # Skip very short city names (< 3 chars) to avoid false positives
            if len(city_lower) < 3:
                continue

            # Search for the city name in the text
            pos = text_lower.find(city_lower)
            if pos != -1:
                end_pos = pos + len(city_lower)

                # Check word boundaries to avoid matching substrings
                # (e.g., "eu" in "heure" or "ur" in "pour")
                is_word_start = pos == 0 or text_lower[pos - 1] in boundary_chars
                is_word_end = end_pos >= len(text_lower) or text_lower[end_pos] in boundary_chars

                if not is_word_start or not is_word_end:
                    continue

                # Check if this position overlaps with already matched positions
                overlap = any(p >= pos and p < end_pos for p in matched_positions)
                if not overlap:
                    # Mark these positions as matched
                    for p in range(pos, end_pos):
                        matched_positions.add(p)

                    matches.append(
                        {
                            "text": city_lower,
                            "matched_city": city.title(),
                            "score": 1.0,
                            "position": pos,  # Use character position
                            "start": pos,
                        }
                    )

        # Stage 2: For words not yet matched, try embedding similarity
        words = text.split()
        word_start = 0

        for word in words:
            clean_word = word.strip(".,;:!?()[]{}\"'")
            if len(clean_word) < 3:  # Skip very short words
                word_start = text.find(word, word_start) + len(word)
                continue

            word_lower = clean_word.lower()
            word_pos = text.find(word, word_start)

            # Skip if already matched or is a stop word
            if word_pos in matched_positions or word_lower in stop_words:
                word_start = word_pos + len(word)
                continue

            # Skip if this word overlaps with any matched region
            overlap = any(
                p >= word_pos and p < word_pos + len(clean_word) for p in matched_positions
            )
            if overlap:
                word_start = word_pos + len(word)
                continue

            # Try embedding similarity (disabled for now - too many false positives)
            # This gives us a pure exact-match baseline
            word_start = word_pos + len(word)

        return matches

    def extract_entities(self, text: str) -> Dict[str, Union[Optional[str], List[str]]]:
        """
        Extract departure and destination using zero-shot embedding matching.

        Args:
            text: Input sentence (e.g., "Je veux aller de Paris à Lyon")

        Returns:
            Dictionary with extracted entities:
            {
                "departure": "Paris",
                "destination": "Lyon",
                "intermediate": []
            }
        """
        result: Dict[str, Union[Optional[str], List[str]]] = {
            "departure": None,
            "destination": None,
            "intermediate": [],
        }

        # Find all station matches
        matches = self._find_station_matches(text)

        if not matches:
            return result

        # Sort by position in text
        matches.sort(key=lambda x: x["position"])

        # Apply heuristics: first = departure, last = destination
        if len(matches) == 1:
            result["destination"] = matches[0]["matched_city"]
        elif len(matches) >= 2:
            result["departure"] = matches[0]["matched_city"]
            result["destination"] = matches[-1]["matched_city"]

            # Middle matches are intermediates
            if len(matches) > 2:
                result["intermediate"] = [m["matched_city"] for m in matches[1:-1]]

        return result


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
