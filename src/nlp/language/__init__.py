"""
Language detection module for Travel Order Resolver.

This module provides language detection implementations:
- RegexLanguageDetector: Rule-based detection for FR/EN/UNKNOWN
"""

from .regex_language import RegexLanguageDetector

__all__ = ["RegexLanguageDetector"]
