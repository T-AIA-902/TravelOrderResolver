"""
Pre-processing module for NLP pipeline.

Provides text cleaning and normalization that runs BEFORE
language detection, intent classification, and entity extraction.

Components:
- STTArtifactFilter: Cleans speech-to-text artifacts ([noise], [music], etc.)
- Preprocessor: Text normalization (unicode, tokenization, accents, etc.)

Pipeline flow:
    Raw Input → PRE (stt_filter, preprocessor) → Language → Intent → Entity → POST
"""

from .preprocessor import Preprocessor, PreprocessorConfig, preprocess, tokenize
from .stt_filter import STTArtifactFilter

__all__ = [
    # STT artifact filtering
    "STTArtifactFilter",
    # Text preprocessing
    "Preprocessor",
    "PreprocessorConfig",
    "preprocess",
    "tokenize",
]
