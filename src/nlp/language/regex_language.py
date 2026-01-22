"""
Regex-based language detector.

Detects whether text is FRENCH, ENGLISH, or UNKNOWN based on
keyword patterns. Optimized for travel-related text in the SNCF context.
"""

import re
from typing import List, Tuple

from ..interfaces import LanguageDetector


class RegexLanguageDetector(LanguageDetector):
    """
    Rule-based language detector using regex patterns.

    Classifies text into:
    - FRENCH: French text (primary language for SNCF)
    - ENGLISH: English text (for back-translation experiments)
    - UNKNOWN: Other languages (ES, DE, IT) or mixed/unclear text

    Note: Spanish, German, and Italian are mapped to UNKNOWN as they
    represent unsupported languages in the FR/EN focused study.
    """

    def __init__(self) -> None:
        """Initialize the regex language detector."""
        self._compile_patterns()

    @property
    def name(self) -> str:
        return "Regex"

    def _compile_patterns(self) -> None:
        """Compile regex patterns for language detection."""
        # French indicators (travel-focused + common words)
        self.french_indicators = [
            # Travel-related
            r"\bje\s+veux\b",
            r"\bje\s+voudrais\b",
            r"\bje\s+souhaite\b",
            r"\baller\s+[aà]\b",
            r"\bpartir\s+de\b",
            r"\bprendre\s+le\s+train\b",
            r"\ben\s+passant\s+par\b",
            r"\bjusqu['\s]?[aà]\b",
            # Common French words
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
            r"\bdu\b",
            r"\bdes\b",
            r"\bbonjour\b",
            r"\bmerci\b",
            r"\bs'il\s+vous\s+pla[iî]t\b",
            r"\bvers\b",
            r"\bpuis\b",
            r"\bdirection\b",
            r"\bqu[ei]\b",
            r"\bc'?est\b",
            r"\bfaut\b",
            r"\bpour\b",
            r"\bdepuis\b",
            r"\bcomment\b",
            r"\bquand\b",
            r"\beuh\b",
            r"\bben\b",
            r"\bbah\b",
        ]

        # English indicators (travel-focused + common words)
        self.english_indicators = [
            # Travel-related
            r"\bI\s+want\s+to\b",
            r"\bI\s+would\s+like\b",
            r"\bI['\s]d\s+like\b",
            r"\bfrom\b.*\bto\b",
            r"\bgoing\s+to\b",
            r"\btravel\s+to\b",
            r"\bticket\s+to\b",
            r"\bget\s+to\b",
            # Common English words
            r"\bI\b",
            r"\bwant\b",
            r"\bwould\b",
            r"\blike\b",
            r"\bplease\b",
            r"\bthank\s+you\b",
            r"\bhow\b",
            r"\bwhat\b",
            r"\bwhere\b",
            r"\bwhen\b",
            r"\bthe\b",
            r"\bthis\b",
            r"\bthat\b",
            r"\bcan\b",
            r"\bcould\b",
            r"\bwill\b",
            r"\bdo\b",
            r"\bdoes\b",
            r"\bam\b",
            r"\bare\b",
            r"\bis\b",
            r"\bmy\b",
            r"\byour\b",
            r"\bme\b",
            r"\byou\b",
            r"\btravel\b",
            r"\btrain\b",
            r"\bstation\b",
        ]

        # Other language indicators (to detect non-FR/EN)
        self.other_language_indicators = [
            # Spanish
            r"\bquiero\b",
            r"\bme\s+gustar[ií]a\b",
            r"\bun\s+billete\b",
            r"\bhola\b",
            r"\bgracias\b",
            r"\bpor\s+favor\b",
            r"\bd[oó]nde\b",
            r"\bcu[aá]ndo\b",
            # German
            r"\bich\s+m[oö]chte\b",
            r"\bein\s+zug\b",
            r"\bfahrkarte\b",
            r"\bguten\s+tag\b",
            r"\bdanke\b",
            r"\bbitte\b",
            r"\bwo\b",
            r"\bwann\b",
            r"\bnach\b",
            # Italian
            r"\bvorrei\b",
            r"\bun\s+biglietto\b",
            r"\bciao\b",
            r"\bbuongiorno\b",
            r"\bper\s+favore\b",
            r"\bdove\b",
            r"\bquando\b",
        ]

        # Compile patterns
        self.french_pattern = re.compile(
            "|".join(self.french_indicators), re.IGNORECASE | re.UNICODE
        )
        self.english_pattern = re.compile(
            "|".join(self.english_indicators), re.IGNORECASE | re.UNICODE
        )
        self.other_pattern = re.compile(
            "|".join(self.other_language_indicators), re.IGNORECASE | re.UNICODE
        )

    def _count_matches(self, text: str, pattern: re.Pattern) -> int:
        """Count the number of pattern matches in text."""
        return len(pattern.findall(text))

    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the input text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (language, confidence) where:
            - language: One of "FRENCH", "ENGLISH", "UNKNOWN"
            - confidence: Float between 0.0 and 1.0
        """
        if not text or len(text.strip()) < 2:
            return ("UNKNOWN", 0.5)

        # Count matches for each language
        fr_count = self._count_matches(text, self.french_pattern)
        en_count = self._count_matches(text, self.english_pattern)
        other_count = self._count_matches(text, self.other_pattern)

        # If other language detected, return UNKNOWN
        if other_count > 0 and other_count >= max(fr_count, en_count):
            return ("UNKNOWN", 0.7)

        # Calculate relative scores
        total = fr_count + en_count + other_count
        if total == 0:
            # No clear indicators, default to UNKNOWN
            return ("UNKNOWN", 0.4)

        # Determine language based on highest count
        if fr_count > en_count:
            confidence = min(0.95, 0.5 + (fr_count / (total + 1)) * 0.5)
            return ("FRENCH", confidence)
        elif en_count > fr_count:
            confidence = min(0.95, 0.5 + (en_count / (total + 1)) * 0.5)
            return ("ENGLISH", confidence)
        elif fr_count == en_count and fr_count > 0:
            # Tie between FR and EN - check for stronger French indicators
            # French travel phrases are more specific
            if re.search(r"\b(je|voudrais|veux|aller|prendre)\b", text, re.IGNORECASE):
                return ("FRENCH", 0.6)
            return ("UNKNOWN", 0.5)

        return ("UNKNOWN", 0.4)

    def detect_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Detect language for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of (language, confidence) tuples
        """
        return [self.detect(text) for text in texts]
