"""
NLP module for travel order resolution.

This module provides natural language processing capabilities
for extracting travel information from French text.
"""

from .models import BaseModel, BaselineRegexModel, Intent, PredictionResult, TravelEntity
from .pipeline import NLPPipeline, PipelineConfig, parse_travel_request
from .preprocessor import Preprocessor, PreprocessorConfig, preprocess, tokenize

__all__ = [
    # Models
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
