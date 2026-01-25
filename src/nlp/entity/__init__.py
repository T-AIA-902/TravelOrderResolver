"""
Entity extraction module.

This module provides entity extractors that identify travel-related
entities (departure, destination, intermediate stops) in text.
"""

from .camembert_entity import CamembertEntityExtractor
from .ministral_entity import MinistralEntityExtractor
from .regex_entity import RegexEntityExtractor
from .spacy_entity import SpacyEntityExtractor

__all__ = [
    "RegexEntityExtractor",
    "SpacyEntityExtractor",
    "CamembertEntityExtractor",
    "MinistralEntityExtractor",
]
