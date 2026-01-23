"""
NLP module for travel order resolution.

This module provides natural language processing capabilities
for extracting travel information from French text.

Architecture:
- LanguageDetector: Detect language of input text (FRENCH, ENGLISH, UNKNOWN)
- IntentClassifier: Classify text as TRIP, NOT_TRIP, UNKNOWN
- EntityExtractor: Extract departure, destination, and intermediate stations
- PostProcessor: Post-process extracted entities (e.g., fuzzy matching)

Available implementations:
- Language: RegexLanguageDetector, LangdetectLanguageDetector
- Intent: RegexIntentClassifier, CamembertIntentClassifier
- Entity: RegexEntityExtractor, SpacyEntityExtractor, CamembertEntityExtractor
- Post: FuzzyPostProcessor
"""

# Entity extractors
from .entity import CamembertEntityExtractor, RegexEntityExtractor, SpacyEntityExtractor

# Intent classifiers
from .intent import CamembertIntentClassifier, RegexIntentClassifier

# Core interfaces
from .interfaces import EntityExtractor, IntentClassifier, LanguageDetector, PostProcessor

# Language detectors
from .language import LangdetectLanguageDetector, RegexLanguageDetector

# Pipeline
from .pipeline import NLPPipeline, PipelineConfig, parse_travel_request

# Post-processors
from .post import FuzzyPostProcessor

# Pre-processors
from .pre import Preprocessor, PreprocessorConfig, STTArtifactFilter, preprocess, tokenize

# Types
from .types import Intent, Language, PredictionResult, TravelEntity

__all__ = [
    # Interfaces
    "LanguageDetector",
    "IntentClassifier",
    "EntityExtractor",
    "PostProcessor",
    # Language detectors
    "RegexLanguageDetector",
    "LangdetectLanguageDetector",
    # Intent classifiers
    "RegexIntentClassifier",
    "CamembertIntentClassifier",
    # Entity extractors
    "RegexEntityExtractor",
    "SpacyEntityExtractor",
    "CamembertEntityExtractor",
    # Post-processors
    "FuzzyPostProcessor",
    # Types
    "Intent",
    "Language",
    "PredictionResult",
    "TravelEntity",
    # Pipeline
    "NLPPipeline",
    "PipelineConfig",
    "parse_travel_request",
    # Pre-processors
    "STTArtifactFilter",
    "Preprocessor",
    "PreprocessorConfig",
    "preprocess",
    "tokenize",
]
