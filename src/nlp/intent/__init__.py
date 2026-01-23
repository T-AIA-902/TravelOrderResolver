"""
Intent classification module.

This module provides intent classifiers that determine whether a text
is a travel request (TRIP), not a travel request (NOT_TRIP), or unknown (UNKNOWN).

Note: Language detection is handled separately by the language module.
"""

from .camembert_intent import CamembertIntentClassifier
from .flant5_intent import FlanT5IntentClassifier
from .regex_intent import RegexIntentClassifier
from .spacy_intent import SpacyIntentClassifier

__all__ = [
    "RegexIntentClassifier",
    "CamembertIntentClassifier",
    "SpacyIntentClassifier",
    "FlanT5IntentClassifier",
]
