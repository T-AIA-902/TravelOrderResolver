"""
Regex-based intent classifier.

Extracted from BaselineRegexModel for modular evaluation.
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
    - NOT_FRENCH: Text is not in French
    - UNKNOWN: Cannot determine intent
    """

    def __init__(self) -> None:
        """Initialize the regex intent classifier."""
        self._compile_patterns()

    @property
    def name(self) -> str:
        return "Regex"

    def _compile_patterns(self) -> None:
        """Compile regex patterns for intent detection."""
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

        # French language indicators
        self.french_indicators = [
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
        ]

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
        french_count = sum(
            1 for pattern in self.french_indicators if re.search(pattern, text, re.IGNORECASE)
        )

        # If French indicators found, it's French
        if french_count >= 1:
            return True

        # If no non-French indicators and text is short, assume French
        return True

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the input text.

        Args:
            text: Input text to classify

        Returns:
            Tuple of (intent_label, confidence) where:
            - intent_label: One of "TRIP", "NOT_TRIP", "NOT_FRENCH", "UNKNOWN"
            - confidence: Float between 0.0 and 1.0
        """
        # Normalize text for processing
        processed = text.lower().strip()

        # Check for non-French first
        if not self._detect_language(text):
            return ("NOT_FRENCH", 0.9)

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
