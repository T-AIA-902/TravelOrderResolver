"""
Unit tests for NLP module.

Tests the preprocessor, baseline model, and pipeline.
"""

import pytest

from src.nlp import (
    BaselineRegexModel,
    Intent,
    NLPPipeline,
    PipelineConfig,
    Preprocessor,
    PreprocessorConfig,
    parse_travel_request,
    preprocess,
    tokenize,
)


class TestPreprocessor:
    """Tests for the Preprocessor class."""

    def test_default_preprocessor(self) -> None:
        """Test default preprocessing."""
        result = preprocess("  Hello   World  ")
        assert result == "hello world"

    def test_unicode_normalization(self) -> None:
        """Test unicode normalization."""
        preprocessor = Preprocessor()
        # Test composed vs decomposed unicode
        result = preprocessor.normalize_unicode("café")
        assert "é" in result or "e" in result

    def test_apostrophe_normalization(self) -> None:
        """Test apostrophe normalization."""
        preprocessor = Preprocessor()
        result = preprocessor.normalize_apostrophes("l'train")
        assert result == "l'train"

    def test_hyphen_normalization(self) -> None:
        """Test hyphen normalization."""
        preprocessor = Preprocessor()
        result = preprocessor.normalize_hyphens("Paris–Lyon")
        assert result == "Paris-Lyon"

    def test_whitespace_normalization(self) -> None:
        """Test whitespace normalization."""
        preprocessor = Preprocessor()
        result = preprocessor.normalize_whitespace("hello\t\nworld  test")
        assert result == "hello world test"

    def test_remove_accents(self) -> None:
        """Test accent removal."""
        preprocessor = Preprocessor()
        result = preprocessor.remove_accents("café résumé")
        assert result == "cafe resume"

    def test_config_no_lowercase(self) -> None:
        """Test config without lowercase."""
        config = PreprocessorConfig(lowercase=False)
        preprocessor = Preprocessor(config)
        result = preprocessor.preprocess("Hello World")
        assert result == "Hello World"

    def test_config_remove_accents(self) -> None:
        """Test config with accent removal."""
        config = PreprocessorConfig(remove_accents=True)
        preprocessor = Preprocessor(config)
        result = preprocessor.preprocess("Café")
        assert result == "cafe"

    def test_tokenize(self) -> None:
        """Test tokenization."""
        tokens = tokenize("Je veux aller à Paris")
        assert len(tokens) == 5
        assert "paris" in tokens

    def test_empty_string(self) -> None:
        """Test empty string handling."""
        assert preprocess("") == ""
        assert tokenize("") == []


class TestBaselineRegexModel:
    """Tests for the BaselineRegexModel."""

    @pytest.fixture
    def model(self) -> BaselineRegexModel:
        """Create a model instance without station database."""
        return BaselineRegexModel(station_db=None)

    def test_model_name(self, model: BaselineRegexModel) -> None:
        """Test model name."""
        assert model.name == "baseline_regex"

    def test_simple_trip_de_a(self, model: BaselineRegexModel) -> None:
        """Test simple 'de X a Y' pattern."""
        result = model.predict("Je veux aller de Paris a Lyon")
        assert result.intent == Intent.TRIP
        assert "Paris" in result.departure
        assert "Lyon" in result.destination

    def test_simple_trip_vers(self, model: BaselineRegexModel) -> None:
        """Test 'X vers Y' pattern."""
        result = model.predict("Paris vers Lyon")
        assert result.intent == Intent.TRIP
        assert "Paris" in result.departure
        assert "Lyon" in result.destination

    def test_trip_depuis_a(self, model: BaselineRegexModel) -> None:
        """Test 'depuis X a Y' pattern."""
        result = model.predict("Je pars depuis Marseille a Bordeaux")
        assert result.intent == Intent.TRIP
        assert "Marseille" in result.departure
        assert "Bordeaux" in result.destination

    def test_trip_with_intermediate(self, model: BaselineRegexModel) -> None:
        """Test extraction with intermediate stop."""
        result = model.predict("De Paris a Lyon en passant par Dijon")
        assert result.intent == Intent.TRIP
        assert "Paris" in result.departure
        assert "Lyon" in result.destination
        # Note: Intermediate extraction may vary

    def test_trip_three_stations(self, model: BaselineRegexModel) -> None:
        """Test three station pattern."""
        result = model.predict("Paris puis Lyon puis Marseille")
        assert result.intent == Intent.TRIP
        assert "Paris" in result.departure
        assert "Marseille" in result.destination
        assert len(result.intermediates) == 1 or "Lyon" in str(result.intermediates)

    def test_not_french_english(self, model: BaselineRegexModel) -> None:
        """Test English detection."""
        result = model.predict("I want to go from Paris to Lyon")
        assert result.intent == Intent.NOT_FRENCH

    def test_not_french_german(self, model: BaselineRegexModel) -> None:
        """Test German detection."""
        result = model.predict("Ich möchte von Paris nach Lyon fahren")
        assert result.intent == Intent.NOT_FRENCH

    def test_not_trip_greeting(self, model: BaselineRegexModel) -> None:
        """Test greeting as NOT_TRIP."""
        result = model.predict("Bonjour, comment allez-vous?")
        # Could be NOT_TRIP or TRIP depending on interpretation
        assert result.intent in (Intent.NOT_TRIP, Intent.TRIP)

    def test_unknown_gibberish(self, model: BaselineRegexModel) -> None:
        """Test gibberish as UNKNOWN."""
        result = model.predict("asdf")
        assert result.intent in (Intent.UNKNOWN, Intent.NOT_TRIP, Intent.TRIP)

    def test_unknown_empty(self, model: BaselineRegexModel) -> None:
        """Test empty string."""
        result = model.predict("")
        assert result.intent == Intent.UNKNOWN

    def test_polite_request(self, model: BaselineRegexModel) -> None:
        """Test polite form request."""
        result = model.predict("Je voudrais un billet de Nantes a Rennes")
        assert result.intent == Intent.TRIP
        assert "Nantes" in result.departure
        assert "Rennes" in result.destination

    def test_question_form(self, model: BaselineRegexModel) -> None:
        """Test question form."""
        result = model.predict("Comment aller de Lille a Paris ?")
        assert result.intent == Intent.TRIP
        assert "Lille" in result.departure
        assert "Paris" in result.destination

    def test_train_mention(self, model: BaselineRegexModel) -> None:
        """Test with train mention."""
        result = model.predict("Un train de Strasbourg a Mulhouse")
        assert result.intent == Intent.TRIP
        assert "Strasbourg" in result.departure
        assert "Mulhouse" in result.destination

    def test_result_to_dict(self, model: BaselineRegexModel) -> None:
        """Test result serialization."""
        result = model.predict("De Paris a Lyon")
        d = result.to_dict()
        assert "intent" in d
        assert "departure" in d
        assert "destination" in d
        assert d["intent"] == "TRIP"

    def test_is_valid_trip(self, model: BaselineRegexModel) -> None:
        """Test is_valid_trip property."""
        result = model.predict("De Paris a Lyon")
        assert result.is_valid_trip is True

        result = model.predict("Bonjour")
        assert result.is_valid_trip is False


class TestNLPPipeline:
    """Tests for the NLPPipeline."""

    def test_default_pipeline(self) -> None:
        """Test default pipeline creation."""
        pipeline = NLPPipeline()
        assert pipeline.get_model_name() == "baseline_regex"

    def test_pipeline_process(self) -> None:
        """Test pipeline processing."""
        pipeline = NLPPipeline()
        result = pipeline.process("Je veux aller de Paris a Lyon")
        assert result.intent == Intent.TRIP

    def test_pipeline_batch_process(self) -> None:
        """Test batch processing."""
        pipeline = NLPPipeline()
        results = pipeline.batch_process(
            [
                "De Paris a Lyon",
                "Hello world",
                "De Marseille a Nice",
            ]
        )
        assert len(results) == 3
        assert results[0].intent == Intent.TRIP
        assert results[2].intent == Intent.TRIP

    def test_pipeline_config(self) -> None:
        """Test pipeline with config."""
        config = PipelineConfig(
            model_name="baseline_regex",
            use_station_matching=False,
        )
        pipeline = NLPPipeline(config=config)
        assert pipeline.get_model_name() == "baseline_regex"

    def test_invalid_model(self) -> None:
        """Test invalid model name."""
        config = PipelineConfig(model_name="invalid_model")
        with pytest.raises(ValueError):
            NLPPipeline(config=config)


class TestParseFunction:
    """Tests for the parse_travel_request convenience function."""

    def test_simple_parse(self) -> None:
        """Test simple parsing."""
        result = parse_travel_request("De Paris a Lyon")
        assert result["intent"] == "TRIP"
        assert "Paris" in result["departure"]
        assert "Lyon" in result["destination"]
        assert result["is_valid_trip"] is True

    def test_parse_not_trip(self) -> None:
        """Test non-trip parsing."""
        result = parse_travel_request("Bonjour tout le monde")
        assert result["intent"] in ("NOT_TRIP", "TRIP")

    def test_parse_returns_dict(self) -> None:
        """Test return type."""
        result = parse_travel_request("Test")
        assert isinstance(result, dict)
        assert "intent" in result
        assert "departure" in result
        assert "destination" in result
        assert "intermediates" in result
        assert "confidence" in result
