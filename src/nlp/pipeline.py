"""
NLP Pipeline for travel order resolution.

This module provides a modular pipeline that orchestrates preprocessing,
language detection, intent classification, and entity extraction.

The pipeline composes modular components:
- LanguageDetector: Detect language of input text
- IntentClassifier: Classify intent as TRIP/NOT_TRIP
- EntityExtractor: Extract departure, destination, intermediates
"""

from dataclasses import dataclass
from typing import Any

from ..data import StationDatabase
from .entity import RegexEntityExtractor
from .intent import RegexIntentClassifier
from .interfaces import EntityExtractor, IntentClassifier, LanguageDetector
from .language import RegexLanguageDetector
from .pre import Preprocessor, PreprocessorConfig
from .types import Intent, Language, PredictionResult


@dataclass
class PipelineConfig:
    """Configuration for NLP pipeline."""

    use_station_matching: bool = True
    preprocessor_config: PreprocessorConfig | None = None


class NLPPipeline:
    """
    Modular NLP pipeline for travel order resolution.

    The pipeline orchestrates:
    1. Text preprocessing
    2. Language detection
    3. Intent classification
    4. Entity extraction (departure, destination, intermediates)
    5. Station matching (optional)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        station_db: StationDatabase | None = None,
        language_detector: LanguageDetector | None = None,
        intent_classifier: IntentClassifier | None = None,
        entity_extractor: EntityExtractor | None = None,
    ):
        """
        Initialize the NLP pipeline.

        Args:
            config: Pipeline configuration.
            station_db: Station database for matching.
            language_detector: Custom language detector (defaults to Regex).
            intent_classifier: Custom intent classifier (defaults to Regex).
            entity_extractor: Custom entity extractor (defaults to Regex).
        """
        self.config = config or PipelineConfig()
        self.station_db = station_db

        # Initialize preprocessor
        prep_config = self.config.preprocessor_config or PreprocessorConfig()
        self.preprocessor = Preprocessor(prep_config)

        # Initialize modular components (allow injection for flexibility)
        self.language_detector = language_detector or RegexLanguageDetector()
        self.intent_classifier = intent_classifier or RegexIntentClassifier()
        self.entity_extractor = entity_extractor or RegexEntityExtractor()

    def process(self, text: str) -> PredictionResult:
        """
        Process text through the pipeline.

        Args:
            text: Input text to process.

        Returns:
            PredictionResult with extracted information.
        """
        # Preprocess text
        processed = self.preprocessor.preprocess(text)

        # Detect language
        lang_str, lang_conf = self.language_detector.detect(text)
        language = Language[lang_str] if lang_str in Language.__members__ else Language.UNKNOWN

        # Classify intent (always run, regardless of language)
        intent_str, intent_conf = self.intent_classifier.classify(text)
        intent = Intent[intent_str] if intent_str in Intent.__members__ else Intent.UNKNOWN

        # Extract entities
        entities = self.entity_extractor.extract(text)
        departure = entities.get("departure") or ""
        destination = entities.get("destination") or ""
        intermediates = entities.get("intermediate") or []

        # Match to known stations if database available
        if self.station_db and self.config.use_station_matching:
            departure = self._match_station(departure) or departure
            destination = self._match_station(destination) or destination
            intermediates = [self._match_station(s) or s for s in intermediates]

        # Downgrade to NOT_TRIP if no stations found for TRIP intent
        if intent == Intent.TRIP and not departure and not destination:
            intent = Intent.NOT_TRIP
            intent_conf = 0.5

        return PredictionResult(
            intent=intent,
            intent_confidence=intent_conf,
            language=language,
            language_confidence=lang_conf,
            departure=departure,
            destination=destination,
            intermediates=intermediates,
            raw_text=text,
            processed_text=processed,
            model_name=self.get_model_name(),
        )

    def _match_station(self, name: str) -> str | None:
        """
        Match extracted name to a known station.

        Args:
            name: Extracted station name.

        Returns:
            Matched station name or None.
        """
        if not self.station_db or not name:
            return name

        results = self.station_db.search_by_name(name)
        if results:
            return results[0].name

        return name

    def batch_process(self, texts: list[str]) -> list[PredictionResult]:
        """
        Process multiple texts through the pipeline.

        Args:
            texts: List of input texts.

        Returns:
            List of PredictionResults.
        """
        return [self.process(text) for text in texts]

    def get_model_name(self) -> str:
        """Get current pipeline model name."""
        return f"pipeline({self.entity_extractor.name})"

    def __repr__(self) -> str:
        return f"NLPPipeline(extractor={self.entity_extractor.name})"


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
        "language": result.language.value,
        "departure": result.departure,
        "destination": result.destination,
        "intermediates": result.intermediates,
        "confidence": result.intent_confidence,
        "language_confidence": result.language_confidence,
        "is_valid_trip": result.is_valid_trip,
    }
