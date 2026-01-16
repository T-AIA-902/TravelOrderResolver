"""
NLP module for travel order resolution.

This module provides natural language processing capabilities
for extracting travel information from French text.

Architecture:
- IntentClassifier: Classify text as TRIP, NOT_TRIP, etc.
- EntityExtractor: Extract departure, destination, and intermediate stations
- PostProcessor: Post-process extracted entities (e.g., fuzzy matching)

Available implementations:
- Intent: RegexIntentClassifier, CamembertIntentClassifier
- Entity: RegexEntityExtractor, SpacyEntityExtractor, CamembertEntityExtractor
- Post: FuzzyPostProcessor
"""

# Entity extractors
from .entity import CamembertEntityExtractor, RegexEntityExtractor, SpacyEntityExtractor

# Intent classifiers
from .intent import CamembertIntentClassifier, RegexIntentClassifier

# Core interfaces
from .interfaces import EntityExtractor, IntentClassifier, PostProcessor

# Legacy models (still available for backwards compatibility)
from .models import BaselineRegexModel, BaseModel, Intent, PredictionResult, TravelEntity

# Pipeline
from .pipeline import NLPPipeline, PipelineConfig, parse_travel_request

# Post-processors
from .post import FuzzyPostProcessor

# Preprocessor
from .preprocessor import Preprocessor, PreprocessorConfig, preprocess, tokenize

__all__ = [
    # Interfaces
    "IntentClassifier",
    "EntityExtractor",
    "PostProcessor",
    # Intent classifiers
    "RegexIntentClassifier",
    "CamembertIntentClassifier",
    # Entity extractors
    "RegexEntityExtractor",
    "SpacyEntityExtractor",
    "CamembertEntityExtractor",
    # Post-processors
    "FuzzyPostProcessor",
    # Legacy models
    "BaseModel",
    "BaselineRegexModel",
    "Intent",
    "PredictionResult",
    "TravelEntity",
    # Pipeline
    "NLPPipeline",
    "PipelineConfig",
    "parse_travel_request",
    # Preprocessor
    "Preprocessor",
    "PreprocessorConfig",
    "preprocess",
    "tokenize",
]
