"""
Baseline regex model for travel order resolution.

This model uses rule-based pattern matching to extract travel information
from French sentences. It serves as a baseline for comparison with
ML-based approaches.
"""

import re

from ...data import StationDatabase, normalize_name
from ..preprocessor import Preprocessor, PreprocessorConfig
from .base_model import BaseModel, Intent, PredictionResult, TravelEntity


class BaselineRegexModel(BaseModel):
    """
    Rule-based model using regex patterns.

    This model uses predefined patterns to:
    1. Detect language (French vs non-French)
    2. Classify intent (trip vs non-trip)
    3. Extract departure, destination, and intermediate stations
    """

    def __init__(
        self,
        station_db: StationDatabase | None = None,
        fuzzy_threshold: float = 0.8,
    ):
        """
        Initialize the baseline model.

        Args:
            station_db: Station database for station matching.
            fuzzy_threshold: Threshold for fuzzy station matching (0-1).
        """
        self.station_db = station_db
        self.fuzzy_threshold = fuzzy_threshold

        # Preprocessor for text normalization
        self.preprocessor = Preprocessor(
            PreprocessorConfig(
                lowercase=True,
                remove_accents=False,
                normalize_whitespace=True,
            )
        )

        # Compile regex patterns
        self._compile_patterns()

    @property
    def name(self) -> str:
        return "baseline_regex"

    def _compile_patterns(self) -> None:
        """Compile regex patterns for extraction."""
        # Patterns for trip detection
        self.trip_indicators = [
            r"\baller\b",
            r"\bvoyager\b",
            r"\bpartir\b",
            r"\bprendre\b.*\btrain\b",
            r"\btrain\b",
            r"\bbillet\b",
            r"\btrajet\b",
            r"\broute\b",
            r"\bdepart\b",
            r"\bdestination\b",
            r"\bdirection\b",
            r"\bjusqu['\s]?[aà]\b",
            r"\bdepuis\b",
            r"\bvers\b",
            r"\bde\b.*\b[aà]\b",
            r"\bpuis\b.*\bpuis\b",  # "X puis Y puis Z" pattern
            r"\bvia\b",
            r"\ben\s+passant\s+par\b",
        ]
        self.trip_pattern = re.compile("|".join(self.trip_indicators), re.IGNORECASE | re.UNICODE)

        # Patterns for non-French detection
        self.non_french_indicators = [
            r"\bI\s+want\b",
            r"\bI\s+would\s+like\b",
            r"\bhow\s+do\s+I\b",
            r"\bplease\b",
            r"\bthank\s+you\b",
            r"\bticket\b",
            r"\btravel\b",
            r"\bfrom\b.*\bto\b",
            r"\bich\s+m[oö]chte\b",
            r"\bquiero\b",
            r"\bvorrei\b",
            r"\bein\s+zug\b",
            r"\bun\s+billete\b",
        ]
        self.non_french_pattern = re.compile(
            "|".join(self.non_french_indicators), re.IGNORECASE | re.UNICODE
        )

        # Patterns for departure/destination extraction
        # Format: "de X a/vers Y"
        de_a_regex = (
            r"(?:de|depuis|au\s+depart\s+de)\s+(.+?)\s+"
            r"(?:[aà]|vers|direction|pour|jusqu['\s]?[aà])\s+(.+?)"
            r"(?:\s*[,.]|\s+(?:en\s+passant|via|demain|ce\s+soir|pour)|$)"
        )
        self.de_a_pattern = re.compile(de_a_regex, re.IGNORECASE | re.UNICODE)

        # Format: "X vers/direction Y"
        self.vers_pattern = re.compile(
            r"^(.+?)\s+(?:vers|direction|-|puis)\s+(.+?)(?:\s*[,.]|$)", re.IGNORECASE | re.UNICODE
        )

        # Format: "a Y depuis/en partant de X"
        a_depuis_regex = r"[aà]\s+(.+?)\s+(?:depuis|en\s+partant\s+de)\s+(.+?)" r"(?:\s*[,.]|$)"
        self.a_depuis_pattern = re.compile(a_depuis_regex, re.IGNORECASE | re.UNICODE)

        # Intermediate station patterns
        via_regex = (
            r"(?:via|en\s+passant\s+par|avec\s+(?:un\s+)?arr[eê]t\s+[aà]|puis)"
            r"\s+(.+?)(?:\s+(?:puis|et)|$)"
        )
        self.via_pattern = re.compile(via_regex, re.IGNORECASE | re.UNICODE)

        # Three station pattern: "X puis Y puis Z"
        self.three_station_pattern = re.compile(
            r"^(.+?)\s+puis\s+(.+?)\s+puis\s+(.+?)$", re.IGNORECASE | re.UNICODE
        )

    def _detect_language(self, text: str) -> bool:
        """
        Detect if text is in French.

        Args:
            text: Input text.

        Returns:
            True if text appears to be French or undetermined.
        """
        # Check for non-French indicators first
        if self.non_french_pattern.search(text):
            return False

        # Check for French indicators
        french_indicators = [
            r"\bje\b",
            r"\bvoudrais\b",
            r"\bveux\b",
            r"\bsouhaite\b",
            r"\baller\b",
            r"\bprendre\b",
            r"\bpartir\b",
            r"\bvoyager\b",
            r"\bun\b",
            r"\bune\b",
            r"\ble\b",
            r"\bla\b",
            r"\bles\b",
            r"\bde\b",
            r"\bdu\b",
            r"\b[aà]\b",
            r"\bpour\b",
            r"\bdepuis\b",
            r"\bbonjour\b",
            r"\bmerci\b",
            r"\bs'il\b",
            r"\bvers\b",
            r"\bpuis\b",
            r"\bdirection\b",
            r"\bdepuis\b",
        ]

        french_count = sum(
            1 for pattern in french_indicators if re.search(pattern, text, re.IGNORECASE)
        )

        # If French indicators found, it's French
        if french_count >= 1:
            return True

        # If no non-French indicators and text is short, assume French
        # (could be just station names)
        return True

    def _detect_intent(self, text: str) -> tuple[Intent, float]:
        """
        Detect user intent.

        Args:
            text: Preprocessed input text.

        Returns:
            Tuple of (Intent, confidence).
        """
        # Check for non-French first
        if not self._detect_language(text):
            return Intent.NOT_FRENCH, 0.9

        # Check for empty or very short text
        if len(text.strip()) < 3:
            return Intent.UNKNOWN, 0.8

        # Check for trip indicators
        if self.trip_pattern.search(text):
            return Intent.TRIP, 0.8

        # Check for simple station pairs (e.g., "Paris Lyon")
        words = text.split()
        if 2 <= len(words) <= 4:
            # Might be just station names
            return Intent.TRIP, 0.5

        # Default to NOT_TRIP
        return Intent.NOT_TRIP, 0.6

    def _extract_stations(self, text: str) -> tuple[str, str, list[str], list[TravelEntity]]:
        """
        Extract station names from text.

        Args:
            text: Input text.

        Returns:
            Tuple of (departure, destination, intermediates, entities).
        """
        departure = ""
        destination = ""
        intermediates: list[str] = []
        entities: list[TravelEntity] = []

        # Try three-station pattern first: "X puis Y puis Z"
        match = self.three_station_pattern.search(text)
        if match:
            departure = match.group(1).strip()
            via = match.group(2).strip()
            destination = match.group(3).strip()
            intermediates = [via]

            dep_norm = normalize_name(departure)
            via_norm = normalize_name(via)
            dest_norm = normalize_name(destination)
            entities = [
                TravelEntity(departure, dep_norm, "DEPARTURE"),
                TravelEntity(via, via_norm, "INTERMEDIATE"),
                TravelEntity(destination, dest_norm, "DESTINATION"),
            ]
            return departure, destination, intermediates, entities

        # Try "de X a Y" pattern
        match = self.de_a_pattern.search(text)
        if match:
            departure = match.group(1).strip()
            destination = match.group(2).strip()

        # Try "a Y depuis X" pattern if no match
        if not departure or not destination:
            match = self.a_depuis_pattern.search(text)
            if match:
                destination = match.group(1).strip()
                departure = match.group(2).strip()

        # Try "X vers Y" pattern if no match
        if not departure or not destination:
            match = self.vers_pattern.search(text)
            if match:
                departure = match.group(1).strip()
                destination = match.group(2).strip()

        # Fallback: try to find two capitalized words/phrases
        if not departure or not destination:
            potential_stations = self._find_potential_stations(text)
            if len(potential_stations) >= 2:
                departure = potential_stations[0]
                destination = potential_stations[-1]
                if len(potential_stations) > 2:
                    intermediates = potential_stations[1:-1]

        # Clean up extracted values
        departure = self._clean_station_name(departure)
        destination = self._clean_station_name(destination)

        # Extract intermediate stations
        via_match = self.via_pattern.search(text)
        if via_match and not intermediates:
            via = self._clean_station_name(via_match.group(1))
            if via and via != departure and via != destination:
                intermediates = [via]

        # Build entities
        if departure:
            entities.append(TravelEntity(departure, normalize_name(departure), "DEPARTURE"))
        if destination:
            entities.append(TravelEntity(destination, normalize_name(destination), "DESTINATION"))
        for via in intermediates:
            entities.append(TravelEntity(via, normalize_name(via), "INTERMEDIATE"))

        return departure, destination, intermediates, entities

    def _find_potential_stations(self, text: str) -> list[str]:
        """
        Find potential station names in text.

        Args:
            text: Input text.

        Returns:
            List of potential station names.
        """
        # Words to filter out
        stopwords = {
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
            "a",
            "à",
            "et",
            "ou",
            "mais",
            "donc",
            "car",
            "ni",
            "or",
            "pour",
            "par",
            "sur",
            "sous",
            "avec",
            "sans",
            "dans",
            "en",
            "aller",
            "prendre",
            "partir",
            "voyager",
            "veux",
            "voudrais",
            "train",
            "billet",
            "trajet",
            "gare",
            "direction",
            "vers",
            "demain",
            "aujourd'hui",
            "soir",
            "matin",
            "ce",
            "cette",
            "comment",
            "quel",
            "quelle",
            "est",
            "sont",
            "avoir",
            "être",
            "puis",
            "via",
            "passant",
            "arret",
            "correspondance",
            "bonjour",
            "merci",
            "s'il",
            "vous",
            "plait",
        }

        # Split on common delimiters
        split_pattern = r"[-,;:]|\bpuis\b|\bvia\b|\bet\b"
        parts = re.split(split_pattern, text, flags=re.IGNORECASE)

        stations = []
        for part in parts:
            cleaned = part.strip()

            # Remove leading prepositions and articles
            prep_pattern = (
                r"^(?:de|du|des|le|la|les|un|une|à|a|au|aux|en|pour|"
                r"vers|direction|depuis|jusqu['\s]?[aà])\s+"
            )
            cleaned = re.sub(prep_pattern, "", cleaned, flags=re.IGNORECASE)

            # Skip if too short or only stopwords
            words = cleaned.lower().split()
            non_stop_words = [w for w in words if w not in stopwords]

            if non_stop_words and len(cleaned) > 1:
                stations.append(cleaned.strip())

        return stations

    def _clean_station_name(self, name: str) -> str:
        """
        Clean up extracted station name.

        Args:
            name: Raw station name.

        Returns:
            Cleaned station name.
        """
        if not name:
            return ""

        name = name.strip()

        # Remove trailing punctuation
        name = re.sub(r"[,.:;!?]+$", "", name)

        # Remove leading articles and prepositions
        name = re.sub(
            r"^(?:le|la|les|un|une|du|de la|de l'|d'|l')\s+", "", name, flags=re.IGNORECASE
        )

        # Remove common trailing words
        name = re.sub(
            r"\s+(?:demain|aujourd'hui|ce soir|s'il vous plait|please|svp)$",
            "",
            name,
            flags=re.IGNORECASE,
        )

        return name.strip()

    def _match_station(self, name: str) -> str | None:
        """
        Match extracted name to a known station.

        Args:
            name: Extracted station name.

        Returns:
            Matched station name or None.
        """
        if not self.station_db or not name:
            return name

        results = self.station_db.search_by_name(name)
        if results:
            return results[0].name

        return name

    def predict(self, text: str) -> PredictionResult:
        """
        Make a prediction on input text.

        Args:
            text: Input text.

        Returns:
            PredictionResult with extracted information.
        """
        # Preprocess text
        processed = self.preprocessor.preprocess(text)

        # Detect intent
        intent, intent_confidence = self._detect_intent(processed)

        # Initialize result
        result = PredictionResult(
            intent=intent,
            intent_confidence=intent_confidence,
            raw_text=text,
            processed_text=processed,
            model_name=self.name,
        )

        # Only extract entities for TRIP intent
        if intent == Intent.TRIP:
            extraction = self._extract_stations(text)
            departure, destination, intermediates, entities = extraction

            # Match to known stations if database available
            if self.station_db:
                departure = self._match_station(departure) or departure
                destination = self._match_station(destination) or destination
                intermediates = [self._match_station(s) or s for s in intermediates]

            result.departure = departure
            result.destination = destination
            result.intermediates = intermediates
            result.entities = entities

            # Downgrade to NOT_TRIP if no stations found
            if not departure and not destination:
                result.intent = Intent.NOT_TRIP
                result.intent_confidence = 0.5

        return result

    def extract_entities(self, text: str) -> dict:
        """
        Extract entities in the format expected by TravelOrderResolver.

        This is an adapter method that wraps predict() to provide a consistent
        interface across all extractors.

        Args:
            text: Input text.

        Returns:
            Dictionary with departure, destination, and intermediate stops.
        """
        result = self.predict(text)
        return {
            "departure": result.departure,
            "destination": result.destination,
            "intermediate": result.intermediates or [],
        }
