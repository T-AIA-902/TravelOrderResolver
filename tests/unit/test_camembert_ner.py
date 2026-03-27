"""
Unit tests for CamemBERT NER model and derived components.

Tests the BIO entity extraction logic (pure logic, no model needed)
and integration tests for the full model (skipped if model absent).
"""

from pathlib import Path

import pytest

from src.nlp.camembert_ner_model import CamembertNERModel


# ---------------------------------------------------------------------------
# Pure logic tests — no model loading required
# ---------------------------------------------------------------------------
class TestExtractEntities:
    """Test BIO tag grouping logic (static method, no model needed)."""

    def test_dep_and_dest(self) -> None:
        """Test extraction of departure and destination."""
        words = ["de", "Paris", "a", "Lyon"]
        labels = ["O", "B-DEP", "O", "B-DEST"]
        result = CamembertNERModel.extract_entities(words, labels)
        assert result["departure"] == "Paris"
        assert result["destination"] == "Lyon"
        assert result["intermediate"] == []

    def test_multi_word_entities(self) -> None:
        """Test multi-word entity grouping via I- continuation tags."""
        words = ["de", "Saint", "Etienne", "a", "La", "Rochelle"]
        labels = ["O", "B-DEP", "I-DEP", "O", "B-DEST", "I-DEST"]
        result = CamembertNERModel.extract_entities(words, labels)
        assert result["departure"] == "Saint Etienne"
        assert result["destination"] == "La Rochelle"

    def test_intermediate_stops(self) -> None:
        """Test extraction of intermediate stops (STEP)."""
        words = ["de", "Paris", "a", "Marseille", "via", "Lyon"]
        labels = ["O", "B-DEP", "O", "B-DEST", "O", "B-STEP"]
        result = CamembertNERModel.extract_entities(words, labels)
        assert result["departure"] == "Paris"
        assert result["destination"] == "Marseille"
        assert result["intermediate"] == ["Lyon"]

    def test_multiple_intermediate_stops(self) -> None:
        """Test extraction of multiple intermediate stops."""
        words = ["de", "Paris", "via", "Lyon", "et", "Dijon", "a", "Marseille"]
        labels = ["O", "B-DEP", "O", "B-STEP", "O", "B-STEP", "O", "B-DEST"]
        result = CamembertNERModel.extract_entities(words, labels)
        assert result["departure"] == "Paris"
        assert result["destination"] == "Marseille"
        assert result["intermediate"] == ["Lyon", "Dijon"]

    def test_multi_word_step(self) -> None:
        """Test multi-word intermediate stop grouping."""
        words = ["via", "Saint", "Pierre", "des", "Corps"]
        labels = ["O", "B-STEP", "I-STEP", "I-STEP", "I-STEP"]
        result = CamembertNERModel.extract_entities(words, labels)
        assert result["departure"] is None
        assert result["destination"] is None
        assert result["intermediate"] == ["Saint Pierre des Corps"]

    def test_all_o_labels(self) -> None:
        """Test that all-O labels produce empty results."""
        words = ["bonjour", "comment", "ca", "va"]
        labels = ["O", "O", "O", "O"]
        result = CamembertNERModel.extract_entities(words, labels)
        assert result["departure"] is None
        assert result["destination"] is None
        assert result["intermediate"] == []

    def test_empty_input(self) -> None:
        """Test that empty input produces empty results."""
        result = CamembertNERModel.extract_entities([], [])
        assert result["departure"] is None
        assert result["destination"] is None
        assert result["intermediate"] == []

    def test_only_departure(self) -> None:
        """Test extraction with only a departure entity."""
        words = ["depuis", "Bordeaux"]
        labels = ["O", "B-DEP"]
        result = CamembertNERModel.extract_entities(words, labels)
        assert result["departure"] == "Bordeaux"
        assert result["destination"] is None

    def test_only_destination(self) -> None:
        """Test extraction with only a destination entity."""
        words = ["vers", "Toulouse"]
        labels = ["O", "B-DEST"]
        result = CamembertNERModel.extract_entities(words, labels)
        assert result["departure"] is None
        assert result["destination"] == "Toulouse"


# ---------------------------------------------------------------------------
# Integration tests — require the fine-tuned model on disk
# ---------------------------------------------------------------------------
_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "camembert-ner-retrain"
_MODEL_AVAILABLE = (_MODEL_PATH / "model.safetensors").exists()


@pytest.fixture(scope="module")
def ner_model():
    """Load the fine-tuned CamemBERT NER model once for all integration tests."""
    if not _MODEL_AVAILABLE:
        pytest.skip("Fine-tuned CamemBERT model not available")
    return CamembertNERModel(model_path=_MODEL_PATH, device="cpu")


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="Fine-tuned model not available")
class TestCamembertEntityExtractor:
    """Integration tests for entity extraction with the real model."""

    def test_extract_paris_lyon(self, ner_model: CamembertNERModel) -> None:
        """Test entity extraction on Paris-Lyon trip."""
        from src.nlp.entity import CamembertEntityExtractor

        extractor = CamembertEntityExtractor(ner_model=ner_model)
        result = extractor.extract("Je voudrais aller de Paris a Lyon")
        assert result["departure"] is not None
        assert result["destination"] is not None

    def test_abc_compliance(self, ner_model: CamembertNERModel) -> None:
        """Test that extractor output conforms to the ABC interface."""
        from src.nlp.entity import CamembertEntityExtractor

        extractor = CamembertEntityExtractor(ner_model=ner_model)
        result = extractor.extract("De Marseille a Toulouse")
        assert "departure" in result
        assert "destination" in result
        assert "intermediate" in result
        assert isinstance(result["intermediate"], list)


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="Fine-tuned model not available")
class TestCamembertIntentClassifier:
    """Integration tests for NER-derived intent classification."""

    def test_trip_classification(self, ner_model: CamembertNERModel) -> None:
        """Test that a travel sentence is classified as TRIP."""
        from src.nlp.intent import CamembertIntentClassifier

        classifier = CamembertIntentClassifier(ner_model=ner_model)
        intent, confidence = classifier.classify("Je veux aller de Paris a Lyon")
        assert intent == "TRIP"
        assert 0.0 <= confidence <= 1.0

    def test_not_trip_classification(self, ner_model: CamembertNERModel) -> None:
        """Test that a non-travel sentence is classified as NOT_TRIP."""
        from src.nlp.intent import CamembertIntentClassifier

        classifier = CamembertIntentClassifier(ner_model=ner_model)
        intent, confidence = classifier.classify("Quel temps fait-il demain")
        assert intent == "NOT_TRIP"
        assert 0.0 <= confidence <= 1.0

    def test_short_text_unknown(self, ner_model: CamembertNERModel) -> None:
        """Test that very short text is classified as UNKNOWN."""
        from src.nlp.intent import CamembertIntentClassifier

        classifier = CamembertIntentClassifier(ner_model=ner_model)
        intent, confidence = classifier.classify("ab")
        assert intent == "UNKNOWN"
        assert confidence == 0.5
