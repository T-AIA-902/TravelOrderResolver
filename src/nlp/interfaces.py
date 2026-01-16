"""
Abstract base classes defining the NLP component interfaces.

This module provides the core interfaces for:
- IntentClassifier: Classify text as TRIP, NOT_TRIP, etc.
- EntityExtractor: Extract departure, destination, and intermediate stations
- PostProcessor: Post-process extracted entities (e.g., fuzzy matching)

All NLP models should implement these interfaces for consistent evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class IntentClassifier(ABC):
    """
    Abstract base class for intent classification.

    Intent classifiers determine whether a text is a travel request (TRIP)
    or something else (NOT_TRIP, NOT_FRENCH, UNKNOWN).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this classifier for display purposes."""
        ...

    @abstractmethod
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
        ...


class EntityExtractor(ABC):
    """
    Abstract base class for entity extraction.

    Entity extractors identify travel-related entities in text:
    - departure: Starting station/city
    - destination: Ending station/city
    - intermediate: List of stops along the way
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this extractor for display purposes."""
        ...

    @abstractmethod
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
        ...


class PostProcessor(ABC):
    """
    Abstract base class for post-processing extracted entities.

    Post-processors refine entity extraction results, for example:
    - Fuzzy matching to normalize station names
    - Validation against a station database
    - Spelling correction
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this post-processor for display purposes."""
        ...

    @abstractmethod
    def process(self, entities: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Post-process extracted entities.

        Args:
            entities: Dictionary with departure, destination, intermediate
            text: Original input text (may be needed for context)

        Returns:
            Dictionary with same structure as input, but refined values
        """
        ...
