"""
Regex-based entity extractor.

Extracted from BaselineRegexModel for modular evaluation.
"""

import re
from typing import Any, Dict, List, Optional

from ..interfaces import EntityExtractor


class RegexEntityExtractor(EntityExtractor):
    """
    Rule-based entity extractor using regex patterns.

    Extracts departure, destination, and intermediate stations
    from travel request text using regex patterns.
    """

    def __init__(self) -> None:
        """Initialize the regex entity extractor."""
        self._compile_patterns()

    @property
    def name(self) -> str:
        return "Regex"

    def _compile_patterns(self) -> None:
        """Compile regex patterns for entity extraction."""
        # Format: "de X a/vers Y"
        de_a_regex = (
            r"(?:de|depuis|au\s+depart\s+de)\s+(.+?)\s+"
            r"(?:[aà]|vers|direction|pour|jusqu['\s]?[aà])\s+(.+?)"
            r"(?:\s*[,.]|\s+(?:en\s+passant|via|demain|ce\s+soir|pour)|$)"
        )
        self.de_a_pattern = re.compile(de_a_regex, re.IGNORECASE | re.UNICODE)

        # Format: "X vers/direction Y"
        self.vers_pattern = re.compile(
            r"^(.+?)\s+(?:vers|direction|-|puis)\s+(.+?)(?:\s*[,.]|$)",
            re.IGNORECASE | re.UNICODE,
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

        # ENGLISH PATTERNS

        # Format: "from X to Y"
        from_to_regex = (
            r"(?:from)\s+(.+?)\s+"
            r"(?:to|towards|for)\s+(.+?)"
            r"(?:\s*[,.]|\s+(?:via|through|stopping|tomorrow|tonight|please)|$)"
        )
        self.from_to_pattern = re.compile(from_to_regex, re.IGNORECASE | re.UNICODE)

        # Format: "X to Y" (simple)
        self.x_to_y_pattern = re.compile(
            r"^(.+?)\s+(?:to)\s+(.+?)(?:\s*[,.]|$)",
            re.IGNORECASE | re.UNICODE,
        )

        # English intermediate patterns
        en_via_regex = (
            r"(?:via|through|stopping at|with a stop at)" r"\s+(.+?)(?:\s+(?:and|then)|$)"
        )
        self.en_via_pattern = re.compile(en_via_regex, re.IGNORECASE | re.UNICODE)

    def _find_potential_stations(self, text: str) -> List[str]:
        """
        Find potential station names in text.

        Args:
            text: Input text.

        Returns:
            List of potential station names.
        """
        # Words to filter out (French + English)
        stopwords = {
            # French pronouns
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
            # French articles/prepositions
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
            # French verbs/travel words
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
            "plait",
            # English pronouns
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "it",
            # English articles/prepositions
            "the",
            "a",
            "an",
            "to",
            "from",
            "for",
            "of",
            "in",
            "on",
            "at",
            "by",
            # English verbs/travel words
            "want",
            "would",
            "like",
            "need",
            "going",
            "travel",
            "go",
            "get",
            "take",
            "ticket",
            "station",
            "trip",
            "journey",
            "tomorrow",
            "today",
            "tonight",
            "please",
            "how",
            "what",
            "which",
            "where",
            "when",
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

        # Remove leading articles and prepositions (French + English)
        name = re.sub(
            r"^(?:le|la|les|un|une|du|de la|de l'|d'|l'|the|a|an)\s+",
            "",
            name,
            flags=re.IGNORECASE,
        )

        # Remove common trailing words (French + English)
        name = re.sub(
            r"\s+(?:demain|aujourd'hui|ce soir|s'il vous plait|svp|"
            r"tomorrow|today|tonight|please)$",
            "",
            name,
            flags=re.IGNORECASE,
        )

        return name.strip()

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
        departure: Optional[str] = None
        destination: Optional[str] = None
        intermediates: List[str] = []

        # Try three-station pattern first: "X puis Y puis Z"
        match = self.three_station_pattern.search(text)
        if match:
            departure = self._clean_station_name(match.group(1))
            via = self._clean_station_name(match.group(2))
            destination = self._clean_station_name(match.group(3))
            intermediates = [via] if via else []
            return {
                "departure": departure,
                "destination": destination,
                "intermediate": intermediates,
            }

        # Try "de X a Y" pattern
        match = self.de_a_pattern.search(text)
        if match:
            departure = self._clean_station_name(match.group(1))
            destination = self._clean_station_name(match.group(2))

        # Try "a Y depuis X" pattern if no match
        if not departure or not destination:
            match = self.a_depuis_pattern.search(text)
            if match:
                destination = self._clean_station_name(match.group(1))
                departure = self._clean_station_name(match.group(2))

        # Try "X vers Y" pattern if no match
        if not departure or not destination:
            match = self.vers_pattern.search(text)
            if match:
                departure = self._clean_station_name(match.group(1))
                destination = self._clean_station_name(match.group(2))

        # ENGLISH PATTERNS

        # Try English "from X to Y" pattern
        if not departure or not destination:
            match = self.from_to_pattern.search(text)
            if match:
                departure = self._clean_station_name(match.group(1))
                destination = self._clean_station_name(match.group(2))

        # Try English "X to Y" pattern
        if not departure or not destination:
            match = self.x_to_y_pattern.search(text)
            if match:
                departure = self._clean_station_name(match.group(1))
                destination = self._clean_station_name(match.group(2))

        # Fallback: try to find two capitalized words/phrases
        if not departure or not destination:
            potential_stations = self._find_potential_stations(text)
            if len(potential_stations) >= 2:
                departure = potential_stations[0]
                destination = potential_stations[-1]
                if len(potential_stations) > 2:
                    intermediates = potential_stations[1:-1]

        # Extract intermediate stations (French + English patterns)
        if not intermediates:
            via_match = self.via_pattern.search(text) or self.en_via_pattern.search(text)
            if via_match:
                via = self._clean_station_name(via_match.group(1))
                if via and via != departure and via != destination:
                    intermediates = [via]

        return {
            "departure": departure,
            "destination": destination,
            "intermediate": intermediates,
        }
