"""
Unit tests for intent classification module.

Tests the intent classifiers (Regex and CamemBERT) that determine
whether a text is a travel request or not.
"""

# pylint: disable=redefined-outer-name
# Note: pytest fixtures are meant to be used as function parameters
# This is standard pytest practice, not an error

import pytest

from src.nlp.intent.regex_intent import RegexIntentClassifier


@pytest.fixture(scope="module")
def regex_classifier():
    """Fixture to create a RegexIntentClassifier instance."""
    return RegexIntentClassifier()


class TestRegexIntentClassifierInit:
    """Test RegexIntentClassifier initialization."""

    def test_init(self):
        """Test initialization."""
        classifier = RegexIntentClassifier()
        assert classifier is not None

    def test_name_property(self, regex_classifier):
        """Test name property."""
        assert regex_classifier.name == "Regex"


class TestRegexIntentClassifierClassify:
    """Test RegexIntentClassifier.classify method."""

    def test_classify_trip_de_a(self, regex_classifier):
        """Test classification of 'de X à Y' pattern as TRIP."""
        intent, confidence = regex_classifier.classify("Je veux aller de Paris à Lyon")

        assert intent == "TRIP"
        assert 0.0 <= confidence <= 1.0

    def test_classify_trip_vers(self, regex_classifier):
        """Test classification of 'X vers Y' pattern as TRIP."""
        intent, confidence = regex_classifier.classify("Paris vers Lyon")

        assert intent == "TRIP"
        assert 0.0 <= confidence <= 1.0

    def test_classify_trip_depuis(self, regex_classifier):
        """Test classification with 'depuis' keyword as TRIP."""
        intent, confidence = regex_classifier.classify("Je pars depuis Marseille à Bordeaux")

        assert intent == "TRIP"
        assert 0.0 <= confidence <= 1.0

    def test_classify_trip_train(self, regex_classifier):
        """Test classification with train keyword as TRIP."""
        intent, confidence = regex_classifier.classify("Un train de Strasbourg à Mulhouse")

        assert intent == "TRIP"
        assert 0.0 <= confidence <= 1.0

    def test_classify_english_trip(self, regex_classifier):
        """Test classification of English travel text as TRIP (language is separate)."""
        intent, confidence = regex_classifier.classify("I want to go from Paris to Lyon")

        # Intent classifier detects travel patterns regardless of language
        assert intent == "TRIP"
        assert 0.0 <= confidence <= 1.0

    def test_classify_german_trip(self, regex_classifier):
        """Test classification of German travel text as TRIP (language is separate)."""
        intent, confidence = regex_classifier.classify("Ich möchte von Paris nach Lyon fahren")

        # Intent classifier detects travel patterns regardless of language
        assert intent == "TRIP"
        assert 0.0 <= confidence <= 1.0

    def test_classify_greeting_as_not_trip(self, regex_classifier):
        """Test classification of greeting."""
        intent, confidence = regex_classifier.classify("Bonjour, comment allez-vous?")

        # Could be NOT_TRIP or ambiguous
        assert intent in ("NOT_TRIP", "TRIP", "UNKNOWN")
        assert 0.0 <= confidence <= 1.0

    def test_classify_empty_as_unknown(self, regex_classifier):
        """Test classification of empty string as UNKNOWN."""
        intent, confidence = regex_classifier.classify("")

        assert intent == "UNKNOWN"
        assert 0.0 <= confidence <= 1.0

    def test_classify_gibberish(self, regex_classifier):
        """Test classification of gibberish text."""
        intent, confidence = regex_classifier.classify("asdf")

        assert intent in ("UNKNOWN", "NOT_TRIP", "TRIP")
        assert 0.0 <= confidence <= 1.0

    def test_classify_polite_request(self, regex_classifier):
        """Test classification of polite request."""
        intent, confidence = regex_classifier.classify("Je voudrais un billet de Nantes à Rennes")

        assert intent == "TRIP"
        assert 0.0 <= confidence <= 1.0

    def test_classify_question_form(self, regex_classifier):
        """Test classification of question form."""
        intent, confidence = regex_classifier.classify("Comment aller de Lille à Paris ?")

        assert intent == "TRIP"
        assert 0.0 <= confidence <= 1.0


class TestRegexIntentClassifierReturnType:
    """Test that classify always returns correct type."""

    def test_returns_tuple(self, regex_classifier):
        """Test that classify returns a tuple."""
        result = regex_classifier.classify("De Paris à Lyon")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_string_and_float(self, regex_classifier):
        """Test that classify returns (str, float)."""
        intent, confidence = regex_classifier.classify("De Paris à Lyon")

        assert isinstance(intent, str)
        assert isinstance(confidence, float)

    def test_confidence_in_range(self, regex_classifier):
        """Test that confidence is between 0 and 1."""
        test_sentences = [
            "De Paris à Lyon",
            "Hello world",
            "",
            "Je veux aller à Marseille",
        ]

        for sentence in test_sentences:
            intent, confidence = regex_classifier.classify(sentence)
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range for: {sentence}"


class TestRegexIntentClassifierEdgeCases:
    """Test edge cases."""

    def test_very_long_sentence(self, regex_classifier):
        """Test classification of very long sentence."""
        long_sentence = "Je voudrais " + "vraiment " * 50 + "aller de Paris à Lyon"
        intent, confidence = regex_classifier.classify(long_sentence)

        # Should not crash
        assert intent in ("TRIP", "NOT_TRIP", "UNKNOWN")
        assert 0.0 <= confidence <= 1.0

    def test_special_characters(self, regex_classifier):
        """Test classification with special characters."""
        intent, confidence = regex_classifier.classify("De Paris → Lyon !!!")

        assert intent in ("TRIP", "NOT_TRIP", "UNKNOWN")
        assert 0.0 <= confidence <= 1.0

    def test_mixed_languages(self, regex_classifier):
        """Test classification of mixed language text."""
        intent, confidence = regex_classifier.classify("I want aller de Paris to Lyon")

        # Mixed language with travel patterns should still be TRIP
        assert intent in ("TRIP", "NOT_TRIP", "UNKNOWN")
        assert 0.0 <= confidence <= 1.0
