"""
Unified NLP module using fine-tuned Ministral 3B.

This module provides a single model that handles the entire NLP pipeline:
- Language Detection
- Intent Classification
- Entity Extraction
"""

from src.nlp.unified.ministral_unified import MinistralUnifiedNLP

__all__ = ["MinistralUnifiedNLP"]
