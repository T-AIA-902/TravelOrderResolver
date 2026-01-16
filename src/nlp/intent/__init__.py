"""
Intent classification module.

This module provides intent classifiers that determine whether a text
is a travel request (TRIP), not a travel request (NOT_TRIP), not French
(NOT_FRENCH), or unknown (UNKNOWN).
"""

from .camembert_intent import CamembertIntentClassifier
from .regex_intent import RegexIntentClassifier

__all__ = [
    "RegexIntentClassifier",
    "CamembertIntentClassifier",
]
