"""
Regex-based intent classifier.

Extracted from BaselineRegexModel for modular evaluation.

Note: Language detection is handled separately by LanguageDetector.
This classifier focuses only on intent (TRIP, NOT_TRIP, UNKNOWN).
"""

import re
from typing import Tuple

from ..interfaces import IntentClassifier


class RegexIntentClassifier(IntentClassifier):
    """
    Rule-based intent classifier using regex patterns.

    Classifies text into:
    - TRIP: Travel request
    - NOT_TRIP: Not a travel request
    - UNKNOWN: Cannot determine intent

    Note: Language detection is now handled separately by LanguageDetector.
    """

    def __init__(self) -> None:
        """Initialize the regex intent classifier."""
        self._compile_patterns()

    @property
    def name(self) -> str:
        return "Regex"

    def _compile_patterns(self) -> None:
        """Compile regex patterns for intent detection."""
        # Patterns for trip detection (multilingual - FR, EN, ES, DE, IT)
        self.trip_indicators = [
            # French
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
            # English
            r"\btravel\b",
            r"\bticket\b",
            r"\bfrom\b.*\bto\b",
            r"\bgoing\s+to\b",
            r"\bget\s+to\b",
            r"\bI\s+want\s+to\s+go\b",
            r"\bI\s+would\s+like\s+to\s+go\b",
            # Spanish
            r"\bquiero\s+ir\b",
            r"\bviajar\b",
            r"\bbillete\b",
            r"\bdesde\b.*\bhasta\b",
            # German
            r"\bfahren\b",
            r"\breisen\b",
            r"\bfahrkarte\b",
            r"\bvon\b.*\bnach\b",
            # Italian
            r"\bvorrei\s+andare\b",
            r"\bviaggiare\b",
            r"\bbiglietto\b",
            r"\bda\b.*\ba\b",
        ]
        self.trip_pattern = re.compile("|".join(self.trip_indicators), re.IGNORECASE | re.UNICODE)

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the input text.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (intent_label, confidence) where:
            - intent_label: One of "TRIP", "NOT_TRIP", "UNKNOWN"
            - confidence: Float between 0.0 and 1.0
        """
        # Normalize text for processing
        processed = text.lower().strip()

        # Check for empty or very short text
        if len(processed) < 3:
            return ("UNKNOWN", 0.8)

        # Check for trip indicators
        if self.trip_pattern.search(processed):
            return ("TRIP", 0.8)

        # Check for simple station pairs (e.g., "Paris Lyon")
        words = processed.split()
        if 2 <= len(words) <= 4:
            # Might be just station names
            return ("TRIP", 0.5)

        # Default to NOT_TRIP
        return ("NOT_TRIP", 0.6)
