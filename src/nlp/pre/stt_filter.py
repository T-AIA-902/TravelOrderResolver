"""
STT Artifact Filter - Pipeline Pre-processing.

Provides a pipeline-level pre-filter to clean speech-to-text artifacts
before any NLP processing (language detection, intent classification, etc.).
Designed to handle real Whisper output in production.

Pipeline flow:
    Raw STT → STT Filter → Language → Intent → Entity
                  ↑
            clean ONCE here
"""

import re
from typing import Tuple


class STTArtifactFilter:
    """
    Modular pre-filter for STT (Speech-to-Text) artifacts.

    Cleans noise markers from text. Only marks as pure artifact
    if nothing meaningful remains after cleaning.

    Example usage:
        filter = STTArtifactFilter()
        cleaned, is_noise = filter.filter("[music] Bonjour je veux")
        # cleaned = "Bonjour je veux", is_noise = False

        cleaned, is_noise = filter.filter("[coupure]")
        # cleaned = "", is_noise = True
    """

    # Comprehensive artifact patterns for production Whisper output
    NOISE_MARKERS = [
        # French Whisper artifacts
        "[coupure]",
        "[bruit]",
        "[musique]",
        "[silence]",
        "[annonce]",
        "[bruit de fond]",
        "[incomprehensible]",
        "*bruit*",
        "[Musique]",
        "[Applaudissements]",
        "[interruption]",
        "[rire]",
        "[rires]",
        "[toux]",
        # English Whisper artifacts
        "[inaudible]",
        "[static]",
        "[noise]",
        "[music]",
        "[cut]",
        "[silence]",
        "[applause]",
        "[???]",
        "[background noise]",
        "[Music]",
        "[Applause]",
        "[laughter]",
        "[cough]",
        # Common placeholders
        "[incompréhensible]",
        "[unintelligible]",
    ]

    # Regex patterns for gibberish/noise
    GIBBERISH_PATTERNS = [
        r"\[.*?\]",  # Any bracketed content [...]
        r"\*[^*]+\*",  # Any asterisk content *...*
        r"\.{3,}",  # Three or more dots
        r"-{3,}",  # Three or more dashes
        r"_{3,}",  # Three or more underscores
        r"…+",  # Unicode ellipsis (one or more)
    ]

    def __init__(self) -> None:
        """Initialize the filter with compiled regex patterns."""
        # Compile the combined pattern for efficiency
        self._noise_pattern = re.compile("|".join(self.GIBBERISH_PATTERNS), re.UNICODE)

    def clean(self, text: str) -> str:
        """
        Remove all artifact markers from text.

        Args:
            text: Input text potentially containing STT artifacts

        Returns:
            Cleaned text with artifacts stripped and whitespace normalized
        """
        if not text:
            return ""

        result = text

        # Remove all known noise markers (exact match, case-insensitive for some)
        for marker in self.NOISE_MARKERS:
            result = result.replace(marker, " ")

        # Remove pattern-based artifacts
        result = self._noise_pattern.sub(" ", result)

        # Normalize whitespace
        result = " ".join(result.split())

        return result.strip()

    def is_pure_artifact(self, text: str) -> bool:
        """
        Check if text is ONLY noise (nothing meaningful left after cleaning).

        Args:
            text: Input text to check

        Returns:
            True only if cleaned text is empty or whitespace
        """
        cleaned = self.clean(text)
        return len(cleaned) == 0

    def filter(self, text: str) -> Tuple[str, bool]:
        """
        Clean text and check if anything meaningful remains.

        Args:
            text: Input text to filter

        Returns:
            Tuple of (cleaned_text, is_pure_noise):
            - cleaned_text: Text with artifacts removed
            - is_pure_noise: True ONLY if nothing meaningful remains

        Examples:
            "[music] Bonjour" -> ("Bonjour", False)  # Continue classifying
            "[coupure]"       -> ("", True)          # Return UNKNOWN
            "Hello [noise] world" -> ("Hello world", False)
        """
        cleaned = self.clean(text)
        is_pure_noise = len(cleaned) == 0
        return (cleaned, is_pure_noise)
