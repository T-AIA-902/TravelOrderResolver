"""
Base model interface for NLP models.

This module defines the abstract interface that all NLP models must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Intent(Enum):
    """Classification of user intent."""

    TRIP = "TRIP"  # Valid travel request
    NOT_TRIP = "NOT_TRIP"  # Not a travel request
    NOT_FRENCH = "NOT_FRENCH"  # Not in French
    UNKNOWN = "UNKNOWN"  # Cannot determine intent


@dataclass
class TravelEntity:
    """
    Extracted travel entity (station name).

    Attributes:
        text: The raw text as found in the input.
        normalized: Normalized form for matching.
        role: Entity role (DEPARTURE, DESTINATION, INTERMEDIATE).
        confidence: Confidence score (0.0 to 1.0).
        start: Start position in original text.
        end: End position in original text.
    """

    text: str
    normalized: str
    role: str  # DEPARTURE, DESTINATION, INTERMEDIATE
    confidence: float = 1.0
    start: int = -1
    end: int = -1


@dataclass
class PredictionResult:
    """
    Result of NLP model prediction.

    Attributes:
        intent: Classified intent.
        intent_confidence: Confidence in intent classification.
        departure: Extracted departure station.
        destination: Extracted destination station.
        intermediates: List of intermediate stops.
        entities: All extracted entities with metadata.
        raw_text: Original input text.
        processed_text: Preprocessed text.
        model_name: Name of the model that made the prediction.
        metadata: Additional model-specific metadata.
    """

    intent: Intent
    intent_confidence: float = 1.0
    departure: str = ""
    destination: str = ""
    intermediates: list[str] = field(default_factory=list)
    entities: list[TravelEntity] = field(default_factory=list)
    raw_text: str = ""
    processed_text: str = ""
    model_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intent": self.intent.value,
            "intent_confidence": self.intent_confidence,
            "departure": self.departure,
            "destination": self.destination,
            "intermediates": self.intermediates,
            "entities": [
                {
                    "text": e.text,
                    "normalized": e.normalized,
                    "role": e.role,
                    "confidence": e.confidence,
                }
                for e in self.entities
            ],
            "model_name": self.model_name,
        }

    @property
    def is_valid_trip(self) -> bool:
        """Check if this is a valid trip with departure and destination."""
        return (
            self.intent == Intent.TRIP
            and bool(self.departure)
            and bool(self.destination)
        )


class BaseModel(ABC):
    """
    Abstract base class for NLP models.

    All models must implement the predict method and provide a name.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def predict(self, text: str) -> PredictionResult:
        """
        Make a prediction on the input text.

        Args:
            text: Input text to analyze.

        Returns:
            PredictionResult with intent and extracted entities.
        """
        pass

    def batch_predict(self, texts: list[str]) -> list[PredictionResult]:
        """
        Make predictions on multiple texts.

        Default implementation calls predict() for each text.
        Override for batch-optimized models.

        Args:
            texts: List of input texts.

        Returns:
            List of PredictionResults.
        """
        return [self.predict(text) for text in texts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
