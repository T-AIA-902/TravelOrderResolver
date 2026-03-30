"""Factory functions for creating NLP models.

This module provides model creation functions that can be imported by:
- CLI (src/evaluation/cli.py)
- Notebooks (notebooks/evaluation.ipynb)
- Dataset scripts (datasets/scripts/)
"""

from pathlib import Path
from typing import Any, Literal, Tuple

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
    models: list[str],
    device: DeviceType = "auto",
    ner_model: Any | None = None,
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

    if "camembert-base" in models:
        from src.nlp.intent.camembert_base_intent import CamembertBaseIntentClassifier

        classifiers.append(("CamemBERT (base)", CamembertBaseIntentClassifier(device=device)))

    if "camembert" in models or "all" in models:
        from src.nlp.intent import CamembertIntentClassifier

        classifiers.append(
            ("CamemBERT", CamembertIntentClassifier(device=device, ner_model=ner_model))
        )

    if "spacy" in models or "all" in models:
        from src.nlp.intent import SpacyIntentClassifier

        classifiers.append(("SpaCy", SpacyIntentClassifier(device=device)))

    if "flant5" in models or "all" in models:
        from src.nlp.intent import FlanT5IntentClassifier

        classifiers.append(("Flan-T5", FlanT5IntentClassifier()))

    if "mistral" in models:
        from src.nlp.intent import MistralIntentClassifier

        classifiers.append(("Mistral (base)", MistralIntentClassifier(adapter_path=None)))
        classifiers.append(("Mistral-LoRA", MistralIntentClassifier()))

    return classifiers


def create_entity_extractors(
    models: list[str],
    device: DeviceType = "auto",
    ner_model: Any | None = None,
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

    if "camembert-base" in models:
        from src.nlp.entity.camembert_base_entity import CamembertBaseEntityExtractor

        extractors.append(("CamemBERT (base)", CamembertBaseEntityExtractor(device=device)))

    if "camembert" in models or "all" in models:
        from src.nlp.entity import CamembertEntityExtractor

        extractors.append(
            ("CamemBERT", CamembertEntityExtractor(device=device, ner_model=ner_model))
        )

    if "flant5" in models or "all" in models:
        from src.nlp.entity import FlanT5EntityExtractor

        extractors.append(("Flan-T5", FlanT5EntityExtractor()))

    if "mistral" in models:
        from src.nlp.entity import MistralEntityExtractor

        extractors.append(("Mistral (base)", MistralEntityExtractor(adapter_path=None)))
        extractors.append(("Mistral-LoRA", MistralEntityExtractor()))

    if "flant5-base" in models:
        from src.nlp.entity import FlanT5EntityExtractor

        extractors.append(("Flan-T5 (base)", FlanT5EntityExtractor(model_name="google/flan-t5-base")))

    return extractors


def create_camembert_components(
    device: DeviceType = "auto",
    model_path: str | Path | None = None,
) -> Tuple[Any, Any]:
    """Create CamemBERT entity extractor and intent classifier sharing one model.

    Avoids loading the 420 MB model twice by creating a single
    CamembertNERModel and passing it to both components.

    Args:
        device: Device for the model ("auto", "cuda", "cpu").
        model_path: Path to the fine-tuned model directory.

    Returns:
        Tuple of (CamembertEntityExtractor, CamembertIntentClassifier).
    """
    from src.nlp.camembert_ner_model import CamembertNERModel
    from src.nlp.entity import CamembertEntityExtractor
    from src.nlp.intent import CamembertIntentClassifier

    ner_model = CamembertNERModel(model_path=model_path, device=device)
    extractor = CamembertEntityExtractor(ner_model=ner_model)
    classifier = CamembertIntentClassifier(ner_model=ner_model)
    return extractor, classifier


def create_fuzzy_post_processor(graph: Any = None) -> Any:
    """Create fuzzy post-processor.

    Args:
        graph: Optional NetworkX graph for main station detection.

    Returns:
        FuzzyPostProcessor instance
    """
    from src.nlp.post import FuzzyPostProcessor

    return FuzzyPostProcessor(graph=graph)
