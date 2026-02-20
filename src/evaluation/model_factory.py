"""Factory functions for creating NLP models.

This module provides model creation functions that can be imported by:
- CLI (src/evaluation/cli.py)
- Notebooks (notebooks/evaluation.ipynb)
- Dataset scripts (datasets/scripts/)
"""

from typing import Any, Literal

DeviceType = Literal["auto", "cuda", "cpu"]


def create_language_detectors(models: list[str]) -> list[tuple[str, Any]]:
    """Create language detectors based on model list.

    Args:
        models: List of model names. Options: "regex", "langdetect", "all"

    Returns:
        List of (name, detector) tuples
    """
    detectors: list[tuple[str, Any]] = []

    if "regex" in models or "all" in models:
        from src.nlp.language import RegexLanguageDetector

        detectors.append(("Regex", RegexLanguageDetector()))

    if "langdetect" in models or "all" in models:
        from src.nlp.language import LangdetectLanguageDetector

        detectors.append(("Langdetect", LangdetectLanguageDetector()))

    return detectors


def create_intent_classifiers(
    models: list[str], device: DeviceType = "auto"
) -> list[tuple[str, Any]]:
    """Create intent classifiers based on model list.

    Args:
        models: List of model names. Options: "regex", "spacy", "camembert", "flant5", "all"
        device: Device for ML models ("auto", "cuda", "cpu")

    Returns:
        List of (name, classifier) tuples
    """
    classifiers: list[tuple[str, Any]] = []

    if "regex" in models or "all" in models:
        from src.nlp.intent import RegexIntentClassifier

        classifiers.append(("Regex", RegexIntentClassifier()))

    if "camembert" in models or "all" in models:
        from src.nlp.intent import CamembertIntentClassifier

        classifiers.append(("CamemBERT", CamembertIntentClassifier(device=device)))

    if "spacy" in models or "all" in models:
        from src.nlp.intent import SpacyIntentClassifier

        classifiers.append(("SpaCy", SpacyIntentClassifier(device=device)))

    if "flant5" in models or "all" in models:
        from src.nlp.intent import FlanT5IntentClassifier

        classifiers.append(("Flan-T5", FlanT5IntentClassifier()))

    return classifiers


def create_entity_extractors(
    models: list[str], device: DeviceType = "auto"
) -> list[tuple[str, Any]]:
    """Create entity extractors based on model list.

    Args:
        models: List of model names. Options: "regex", "spacy", "camembert", "flant5", "all"
        device: Device for ML models ("auto", "cuda", "cpu")

    Returns:
        List of (name, extractor) tuples
    """
    extractors: list[tuple[str, Any]] = []

    if "regex" in models or "all" in models:
        from src.nlp.entity import RegexEntityExtractor

        extractors.append(("Regex", RegexEntityExtractor()))

    if "spacy" in models or "all" in models:
        from src.nlp.entity import SpacyEntityExtractor

        extractors.append(("SpaCy", SpacyEntityExtractor(device=device)))

    if "camembert" in models or "all" in models:
        from src.nlp.entity import CamembertEntityExtractor

        extractors.append(("CamemBERT", CamembertEntityExtractor(device=device)))

    if "flant5" in models or "all" in models:
        from src.nlp.entity import FlanT5EntityExtractor

        extractors.append(("Flan-T5", FlanT5EntityExtractor()))

    return extractors


def create_fuzzy_post_processor() -> Any:
    """Create fuzzy post-processor.

    Returns:
        FuzzyPostProcessor instance
    """
    from src.nlp.post import FuzzyPostProcessor

    return FuzzyPostProcessor()
