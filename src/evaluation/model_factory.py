"""Factory functions for creating NLP models.

This module provides model creation functions that can be imported by:
- CLI (src/evaluation/cli.py)
- Notebooks (notebooks/evaluation.ipynb)
- Dataset scripts (datasets/scripts/)
"""

from typing import Any, Literal, Optional

DeviceType = Literal["auto", "cuda", "cpu"]


def create_unified_nlp(
    adapter_path: str = "models/ministral-unified-lora",
    device: DeviceType = "auto",
) -> Any:
    """Create unified Ministral NLP model.

    The unified model handles language detection, intent classification,
    and entity extraction in a single inference pass.

    Args:
        adapter_path: Path to the LoRA adapter directory
        device: Device for model ("auto", "cuda", "cpu")

    Returns:
        MinistralUnifiedNLP instance
    """
    from src.nlp.unified import MinistralUnifiedNLP

    return MinistralUnifiedNLP(adapter_path=adapter_path)


def create_language_detectors(
    models: list[str],
    unified_adapter_path: Optional[str] = None,
) -> list[tuple[str, Any]]:
    """Create language detectors based on model list.

    Args:
        models: List of model names. Options: "regex", "langdetect", "ministral-unified", "all"
        unified_adapter_path: Path to unified adapter (for ministral-unified)

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

    if "ministral-unified" in models:
        from src.nlp.unified import MinistralUnifiedNLP

        adapter = unified_adapter_path or "models/ministral-unified-lora"
        detectors.append(("MinistralUnified", MinistralUnifiedNLP(adapter_path=adapter)))

    return detectors


def create_intent_classifiers(
    models: list[str],
    device: DeviceType = "auto",
    unified_adapter_path: Optional[str] = None,
) -> list[tuple[str, Any]]:
    """Create intent classifiers based on model list.

    Args:
        models: List of model names.
            Options: "regex", "spacy", "camembert", "ministral", "ministral-unified", "all"
        device: Device for ML models ("auto", "cuda", "cpu")
        unified_adapter_path: Path to unified adapter (for ministral-unified)

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

    if "ministral" in models:
        from src.nlp.intent import MinistralIntentClassifier

        classifiers.append(("Ministral", MinistralIntentClassifier(device=device)))

    if "ministral-unified" in models:
        from src.nlp.unified import MinistralUnifiedNLP

        adapter = unified_adapter_path or "models/ministral-unified-lora"
        classifiers.append(("MinistralUnified", MinistralUnifiedNLP(adapter_path=adapter)))

    return classifiers


def create_entity_extractors(
    models: list[str],
    device: DeviceType = "auto",
    unified_adapter_path: Optional[str] = None,
) -> list[tuple[str, Any]]:
    """Create entity extractors based on model list.

    Args:
        models: List of model names.
            Options: "regex", "spacy", "camembert", "ministral", "ministral-unified", "all"
        device: Device for ML models ("auto", "cuda", "cpu")
        unified_adapter_path: Path to unified adapter (for ministral-unified)

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

    if "ministral" in models:
        from src.nlp.entity import MinistralEntityExtractor

        extractors.append(("Ministral", MinistralEntityExtractor(device=device)))

    if "ministral-unified" in models:
        from src.nlp.unified import MinistralUnifiedNLP

        adapter = unified_adapter_path or "models/ministral-unified-lora"
        extractors.append(("MinistralUnified", MinistralUnifiedNLP(adapter_path=adapter)))

    return extractors


def create_fuzzy_post_processor() -> Any:
    """Create fuzzy post-processor.

    Returns:
        FuzzyPostProcessor instance
    """
    from src.nlp.post import FuzzyPostProcessor

    return FuzzyPostProcessor()
