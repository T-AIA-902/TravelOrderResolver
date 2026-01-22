"""
Language detection module for Travel Order Resolver.

This module provides language detection implementations:
- RegexLanguageDetector: Rule-based detection for FR/EN/UNKNOWN
- LangdetectLanguageDetector: Library-based detection using langdetect
"""

from .langdetect_language import LangdetectLanguageDetector
from .regex_language import RegexLanguageDetector

__all__ = ["RegexLanguageDetector", "LangdetectLanguageDetector"]
