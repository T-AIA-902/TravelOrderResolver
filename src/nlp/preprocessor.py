"""
Text preprocessor for NLP pipeline.

This module provides text normalization and preprocessing utilities
for the travel order resolution system.
"""

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class PreprocessorConfig:
    """Configuration for text preprocessing."""

    lowercase: bool = True
    remove_accents: bool = False  # Keep accents by default for French
    normalize_whitespace: bool = True
    remove_punctuation: bool = False
    normalize_apostrophes: bool = True
    normalize_hyphens: bool = True
    strip_extra_spaces: bool = True


class Preprocessor:
    """
    Text preprocessor for NLP input.

    Handles normalization, cleaning, and tokenization of French text
    for travel order resolution.
    """

    def __init__(self, config: PreprocessorConfig | None = None):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration. Uses defaults if None.
        """
        self.config = config or PreprocessorConfig()

    def normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters.

        Args:
            text: Input text.

        Returns:
            Unicode-normalized text (NFC form).
        """
        return unicodedata.normalize("NFC", text)

    def remove_accents(self, text: str) -> str:
        """
        Remove accents from text.

        Args:
            text: Input text.

        Returns:
            Text without accents.
        """
        nfkd = unicodedata.normalize("NFD", text)
        return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")

    def normalize_apostrophes(self, text: str) -> str:
        """
        Normalize various apostrophe characters to standard form.

        Args:
            text: Input text.

        Returns:
            Text with normalized apostrophes.
        """
        # Replace various apostrophe characters with standard apostrophe
        apostrophes = ["'", "'", "ʼ", "ˈ", "´", "`"]
        for apo in apostrophes:
            text = text.replace(apo, "'")
        return text

    def normalize_hyphens(self, text: str) -> str:
        """
        Normalize various hyphen/dash characters.

        Args:
            text: Input text.

        Returns:
            Text with normalized hyphens.
        """
        # Replace various dash characters with standard hyphen
        dashes = ["–", "—", "−", "‐", "‑", "⁃"]
        for dash in dashes:
            text = text.replace(dash, "-")
        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters.

        Args:
            text: Input text.

        Returns:
            Text with normalized whitespace.
        """
        # Replace various whitespace with standard space
        text = re.sub(r"[\t\n\r\f\v]+", " ", text)
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)
        return text.strip()

    def remove_punctuation(self, text: str, keep: str = "'-") -> str:
        """
        Remove punctuation except specified characters.

        Args:
            text: Input text.
            keep: Punctuation characters to keep.

        Returns:
            Text with punctuation removed.
        """
        # Build pattern to remove punctuation except 'keep' characters
        pattern = f"[^\\w\\s{re.escape(keep)}]"
        return re.sub(pattern, "", text)

    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline.

        Args:
            text: Input text.

        Returns:
            Preprocessed text.
        """
        if not text:
            return ""

        result = text

        # Unicode normalization (always applied)
        result = self.normalize_unicode(result)

        # Normalize apostrophes
        if self.config.normalize_apostrophes:
            result = self.normalize_apostrophes(result)

        # Normalize hyphens
        if self.config.normalize_hyphens:
            result = self.normalize_hyphens(result)

        # Remove accents (optional)
        if self.config.remove_accents:
            result = self.remove_accents(result)

        # Lowercase (optional)
        if self.config.lowercase:
            result = result.lower()

        # Remove punctuation (optional)
        if self.config.remove_punctuation:
            result = self.remove_punctuation(result)

        # Normalize whitespace
        if self.config.normalize_whitespace:
            result = self.normalize_whitespace(result)

        # Strip extra spaces
        if self.config.strip_extra_spaces:
            result = result.strip()

        return result

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into words.

        Simple whitespace tokenization with handling of punctuation.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        # Preprocess first
        processed = self.preprocess(text)

        # Split on whitespace
        tokens = processed.split()

        return tokens

    def get_char_spans(self, text: str, tokens: list[str]) -> list[tuple[int, int]]:
        """
        Get character spans for tokens in original text.

        Args:
            text: Original text.
            tokens: List of tokens.

        Returns:
            List of (start, end) tuples for each token.
        """
        spans = []
        current_pos = 0
        text_lower = text.lower() if self.config.lowercase else text

        for token in tokens:
            # Find token in remaining text
            start = text_lower.find(token, current_pos)
            if start == -1:
                # Token not found - use approximate position
                spans.append((current_pos, current_pos + len(token)))
            else:
                end = start + len(token)
                spans.append((start, end))
                current_pos = end

        return spans


# Default preprocessor instance
default_preprocessor = Preprocessor()


def preprocess(text: str) -> str:
    """Preprocess text using default settings."""
    return default_preprocessor.preprocess(text)


def tokenize(text: str) -> list[str]:
    """Tokenize text using default settings."""
    return default_preprocessor.tokenize(text)
