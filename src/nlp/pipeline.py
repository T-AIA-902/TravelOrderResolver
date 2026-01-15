"""
NLP Pipeline for travel order resolution.

This module provides a modular pipeline that orchestrates preprocessing,
intent classification, and entity extraction.
"""

from dataclasses import dataclass
from typing import Any

from ..data import StationDatabase
from .models import BaselineRegexModel, BaseModel, PredictionResult
from .preprocessor import Preprocessor, PreprocessorConfig


@dataclass
class PipelineConfig:
    """Configuration for NLP pipeline."""

    model_name: str = "baseline_regex"
    use_station_matching: bool = True
    preprocessor_config: PreprocessorConfig | None = None


class NLPPipeline:
    """
    Modular NLP pipeline for travel order resolution.

    The pipeline orchestrates:
    1. Text preprocessing
    2. Intent classification
    3. Entity extraction (departure, destination, intermediates)
    4. Station matching (optional)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        station_db: StationDatabase | None = None,
    ):
        """
        Initialize the NLP pipeline.

        Args:
            config: Pipeline configuration.
            station_db: Station database for matching.
        """
        self.config = config or PipelineConfig()
        self.station_db = station_db

        # Initialize preprocessor
        prep_config = self.config.preprocessor_config or PreprocessorConfig()
        self.preprocessor = Preprocessor(prep_config)

        # Initialize model
        self.model = self._create_model()

    def _create_model(self) -> BaseModel:
        """Create the NLP model based on configuration."""
        model_name = self.config.model_name.lower()

        if model_name == "baseline_regex":
            station = self.station_db if self.config.use_station_matching else None
            return BaselineRegexModel(station_db=station)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def process(self, text: str) -> PredictionResult:
        """
        Process text through the pipeline.

        Args:
            text: Input text to process.

        Returns:
            PredictionResult with extracted information.
        """
        return self.model.predict(text)

    def batch_process(self, texts: list[str]) -> list[PredictionResult]:
        """
        Process multiple texts through the pipeline.

        Args:
            texts: List of input texts.

        Returns:
            List of PredictionResults.
        """
        results: list[PredictionResult] = self.model.batch_predict(texts)
        return results

    def set_model(self, model: BaseModel) -> None:
        """
        Set a custom model.

        Args:
            model: Model instance implementing BaseModel.
        """
        self.model = model

    def get_model_name(self) -> str:
        """Get current model name."""
        name: str = self.model.name
        return name

    def __repr__(self) -> str:
        return f"NLPPipeline(model={self.model.name})"


# Convenience function for simple usage
def parse_travel_request(text: str, station_db: StationDatabase | None = None) -> dict[str, Any]:
    """
    Parse a travel request from text.

    Simple interface for quick usage without explicit pipeline setup.

    Args:
        text: Input text to parse.
        station_db: Optional station database for matching.

    Returns:
        Dictionary with parsed information.

    Example:
        >>> result = parse_travel_request("Je veux aller de Paris a Lyon")
        >>> print(result)
        {'intent': 'TRIP', 'departure': 'Paris', 'destination': 'Lyon', ...}
    """
    pipeline = NLPPipeline(station_db=station_db)
    result = pipeline.process(text)

    return {
        "intent": result.intent.value,
        "departure": result.departure,
        "destination": result.destination,
        "intermediates": result.intermediates,
        "confidence": result.intent_confidence,
        "is_valid_trip": result.is_valid_trip,
    }
