"""
Intent classification module.

This module provides intent classifiers that determine whether a text
is a travel request (TRIP), not a travel request (NOT_TRIP), or unknown (UNKNOWN).

Note: Language detection is handled separately by the language module.
"""

from .camembert_intent import CamembertIntentClassifier
from .regex_intent import RegexIntentClassifier

__all__ = [
    "RegexIntentClassifier",
    "CamembertIntentClassifier",
]
