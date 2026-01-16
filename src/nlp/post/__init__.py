"""
Post-processing module.

This module provides post-processors that refine entity extraction results,
such as fuzzy matching to normalize station names.
"""

from .fuzzy_matcher import FuzzyPostProcessor

__all__ = [
    "FuzzyPostProcessor",
]
