"""
Type definitions for NLP module.

This module defines the core types used across the NLP pipeline.
Moved from models/base_model.py during modularization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Intent(Enum):
    """Classification of user intent."""

    TRIP = "TRIP"  # Valid travel request
    NOT_TRIP = "NOT_TRIP"  # Not a travel request
    UNKNOWN = "UNKNOWN"  # Cannot determine intent


class Language(Enum):
    """Classification of text language."""

    FRENCH = "FRENCH"  # French text (primary language for SNCF)
    ENGLISH = "ENGLISH"  # English text (for back-translation experiments)
    UNKNOWN = "UNKNOWN"  # Other languages or unclear


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
    Result of NLP pipeline prediction.

    Attributes:
        intent: Classified intent.
        intent_confidence: Confidence in intent classification.
        language: Detected language.
        language_confidence: Confidence in language detection.
        departure: Extracted departure station.
        destination: Extracted destination station.
        intermediates: List of intermediate stops.
        entities: All extracted entities with metadata.
        raw_text: Original input text.
        processed_text: Preprocessed text.
        model_name: Name of the model/pipeline that made the prediction.
        metadata: Additional model-specific metadata.
    """

    intent: Intent
    intent_confidence: float = 1.0
    language: Language = Language.UNKNOWN
    language_confidence: float = 0.5
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
            "language": self.language.value,
            "language_confidence": self.language_confidence,
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
        return self.intent == Intent.TRIP and bool(self.departure) and bool(self.destination)
