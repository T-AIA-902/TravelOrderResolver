"""
Langdetect-based language detector.

Uses the langdetect library for robust language detection.
"""

from typing import List, Tuple

from ..interfaces import LanguageDetector


class LangdetectLanguageDetector(LanguageDetector):
    """
    Language detector using the langdetect library.

    Maps langdetect ISO codes to our categories:
    - 'fr' -> FRENCH
    - 'en' -> ENGLISH
    - others -> UNKNOWN
    """

    def __init__(self) -> None:
        """Initialize the langdetect detector."""
        try:
            import langdetect

            # Set seed for reproducibility
            langdetect.DetectorFactory.seed = 0
        except ImportError:
            raise ImportError("langdetect not available. Install with: pip install langdetect")

    @property
    def name(self) -> str:
        return "Langdetect"

    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect language using langdetect.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (language, confidence) where:
            - language: One of "FRENCH", "ENGLISH", "UNKNOWN"
            - confidence: Float between 0.0 and 1.0
        """
        import langdetect
        from langdetect import LangDetectException

        if not text or len(text.strip()) < 3:
            return ("UNKNOWN", 0.5)

        try:
            # Get language probabilities
            probs = langdetect.detect_langs(text)
            if not probs:
                return ("UNKNOWN", 0.4)

            top = probs[0]
            lang_code = top.lang
            confidence = top.prob

            # Map to our categories
            if lang_code == "fr":
                return ("FRENCH", confidence)
            elif lang_code == "en":
                return ("ENGLISH", confidence)
            else:
                return ("UNKNOWN", confidence)

        except LangDetectException:
            return ("UNKNOWN", 0.3)

    def detect_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Detect language for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of (language, confidence) tuples
        """
        return [self.detect(text) for text in texts]
