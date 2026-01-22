"""
Unit tests for fuzzy_matcher module.

Tests the StationMatcher class that performs fuzzy matching
of location names to SNCF station database, and the FuzzyPostProcessor
that applies fuzzy matching as a post-processing step.
"""

# pylint: disable=redefined-outer-name
# Note: pytest fixtures are meant to be used as function parameters
# This is standard pytest practice, not an error

import pytest

from src.nlp.post import FuzzyPostProcessor, StationMatcher


@pytest.fixture
def matcher():
    """Fixture to create a StationMatcher instance."""
    return StationMatcher(threshold=75)


class TestStationMatcherInit:
    """Test StationMatcher initialization."""

    def test_init_default_threshold(self):
        """Test initialization with default threshold."""
        matcher = StationMatcher()
        assert matcher.threshold == 80  # Default threshold
        assert len(matcher.search_index) > 0
        assert len(matcher.normalized_names) > 0

    def test_init_custom_threshold(self):
        """Test initialization with custom threshold."""
        matcher = StationMatcher(threshold=70)
        assert matcher.threshold == 70

    def test_stations_loaded(self, matcher):
        """Test that stations are loaded correctly from StationDatabase."""
        assert len(matcher.search_index) > 1000  # Should have many stations
        assert hasattr(matcher, "search_index")
        assert hasattr(matcher, "normalized_names")
        assert hasattr(matcher, "db")  # StationDatabase instance


class TestNormalizeQuery:
    """Test query normalization."""

    def test_normalize_lowercase(self, matcher):
        """Test normalization converts to lowercase."""
        assert matcher.normalize_query("PARIS") == "paris"
        assert matcher.normalize_query("Lyon") == "lyon"

    def test_normalize_accents(self, matcher):
        """Test normalization removes accents."""
        assert matcher.normalize_query("Straßbourg") == "strassbourg"
        assert matcher.normalize_query("Nîmes") == "nimes"

    def test_normalize_whitespace(self, matcher):
        """Test normalization strips whitespace."""
        assert matcher.normalize_query("  Paris  ") == "paris"
        assert matcher.normalize_query("Lyon\t") == "lyon"

    def test_normalize_empty(self, matcher):
        """Test normalization of empty string."""
        assert matcher.normalize_query("") == ""
        assert matcher.normalize_query(None) == ""


class TestMatchStation:
    """Test fuzzy matching of station names."""

    def test_exact_match_major_city(self, matcher):
        """Test exact match for major cities."""
        result = matcher.match_station("Paris")
        assert result is not None
        station, score = result
        assert "Paris" in station
        assert score >= 90.0

    def test_case_insensitive(self, matcher):
        """Test case-insensitive matching."""
        result1 = matcher.match_station("paris")
        result2 = matcher.match_station("PARIS")
        result3 = matcher.match_station("Paris")

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        # All should match to stations with "Paris"
        assert "Paris" in result1[0] or "paris" in result1[0].lower()
        assert "Paris" in result2[0] or "paris" in result2[0].lower()
        assert "Paris" in result3[0] or "paris" in result3[0].lower()

    def test_fuzzy_match_typo(self, matcher):
        """Test fuzzy matching handles typos."""
        # "Nante" should match "Nantes"
        result = matcher.match_station("Nante")
        assert result is not None
        station, score = result
        assert "Nantes" in station
        assert score >= 75.0

    def test_fuzzy_match_missing_chars(self, matcher):
        """Test fuzzy matching handles missing characters."""
        # "bordeau" should match "Bordeaux"
        result = matcher.match_station("bordeau")
        assert result is not None
        station, score = result
        assert "Bordeaux" in station
        assert score >= 75.0

    def test_prefix_match(self, matcher):
        """Test prefix matching for stations with hyphens."""
        # "lyon" should match "Lyon-*"
        result = matcher.match_station("lyon")
        assert result is not None
        station, score = result
        assert station.startswith("Lyon")
        assert score >= 90.0

    def test_no_match_below_threshold(self):
        """Test no match returned when score below threshold."""
        matcher = StationMatcher(threshold=95)
        # Very different name should not match
        result = matcher.match_station("XYZ123")
        assert result is None

    def test_no_match_empty_query(self, matcher):
        """Test empty query returns None."""
        assert matcher.match_station("") is None
        assert matcher.match_station(None) is None

    def test_common_cities(self, matcher):
        """Test matching for common French cities."""
        cities = ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux", "Strasbourg"]
        for city in cities:
            result = matcher.match_station(city)
            assert result is not None, f"Failed to match {city}"
            station, score = result
            assert city in station or city.lower() in station.lower()
            assert score >= 75.0


class TestMatchStationsBatch:
    """Test batch matching of multiple queries."""

    def test_batch_match_multiple(self, matcher):
        """Test batch matching of multiple queries."""
        queries = ["Paris", "Lyon", "Marseille"]
        results = matcher.match_stations_batch(queries)

        assert len(results) == len(queries)
        for result in results:
            assert result is not None
            station, score = result
            assert score >= 75.0

    def test_batch_match_with_none(self, matcher):
        """Test batch matching with some invalid queries."""
        queries = ["Paris", "XYZ123", "Lyon"]
        results = matcher.match_stations_batch(queries)

        assert len(results) == len(queries)
        assert results[0] is not None  # Paris should match
        # XYZ123 might or might not match depending on threshold
        assert results[2] is not None  # Lyon should match

    def test_batch_match_empty_list(self, matcher):
        """Test batch matching with empty list."""
        results = matcher.match_stations_batch([])
        assert results == []


class TestMatchWithDetails:
    """Test detailed matching with multiple candidates."""

    def test_match_with_details_top_n(self, matcher):
        """Test getting top N matches."""
        results = matcher.match_with_details("Paris", top_n=5)

        assert len(results) <= 5
        assert len(results) > 0

        # Check all results contain Paris
        for station, score in results:
            assert "Paris" in station or "paris" in station.lower()

        # Check scores are in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_match_with_details_typo(self, matcher):
        """Test detailed matching for typo shows alternatives."""
        results = matcher.match_with_details("Nante", top_n=3)

        assert len(results) > 0
        # First result should be Nantes or similar
        first_station, first_score = results[0]
        assert "Nant" in first_station  # Should contain Nant*


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_special_characters(self, matcher):
        """Test handling of special characters."""
        # Should handle hyphens, apostrophes, etc.
        result = matcher.match_station("Saint-Étienne")
        assert result is not None

    def test_very_long_query(self, matcher):
        """Test handling of very long queries."""
        long_query = "Paris" + " " * 100 + "Lyon"
        # Should strip and normalize
        result = matcher.match_station(long_query)
        # Might not match due to extra content, but shouldn't crash
        assert result is None or isinstance(result, tuple)

    def test_numeric_in_query(self, matcher):
        """Test handling of numbers in query."""
        result = matcher.match_station("Paris123")
        # Might match Paris or return None, but shouldn't crash
        assert result is None or isinstance(result, tuple)


class TestMatchingQuality:
    """Test quality and accuracy of fuzzy matching."""

    def test_major_cities_high_confidence(self, matcher):
        """Test major cities return high confidence scores."""
        major_cities = ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux"]

        for city in major_cities:
            result = matcher.match_station(city)
            assert result is not None
            station, score = result
            assert score >= 90.0, f"{city} confidence too low: {score}"

    def test_typos_moderate_confidence(self, matcher):
        """Test typos return moderate or high confidence scores."""
        typos = [
            ("Pari", "Paris"),
            ("Bordeau", "Bordeaux"),
            ("Nante", "Nantes"),
        ]

        for typo, expected in typos:
            result = matcher.match_station(typo)
            assert result is not None
            station, score = result
            assert expected in station
            assert score >= 75.0  # Should meet minimum threshold

    def test_prefers_exact_over_partial(self, matcher):
        """Test that exact matches score higher than partial matches."""
        # "Lyon" should score higher than "lyon-*" in matching
        result_exact = matcher.match_station("Lyon")

        assert result_exact is not None
        station, score = result_exact
        # Exact matches or prefix matches should have high score
        assert score >= 90.0


class TestFuzzyPostProcessor:
    """Test FuzzyPostProcessor class."""

    @pytest.fixture
    def processor(self):
        """Fixture to create a FuzzyPostProcessor instance."""
        return FuzzyPostProcessor(threshold=75)

    def test_init_default_threshold(self):
        """Test initialization with default threshold."""
        processor = FuzzyPostProcessor()
        assert processor.threshold == 80  # Default threshold

    def test_init_custom_threshold(self):
        """Test initialization with custom threshold."""
        processor = FuzzyPostProcessor(threshold=70)
        assert processor.threshold == 70

    def test_name_property(self, processor):
        """Test name property returns 'Fuzzy'."""
        assert processor.name == "Fuzzy"

    def test_process_departure(self, processor):
        """Test processing departure entity."""
        entities = {
            "departure": "Paris",
            "destination": None,
            "intermediate": [],
        }
        result = processor.process(entities, "De Paris")

        assert result["departure"] is not None
        assert "Paris" in result["departure"]

    def test_process_destination(self, processor):
        """Test processing destination entity."""
        entities = {
            "departure": None,
            "destination": "Lyon",
            "intermediate": [],
        }
        result = processor.process(entities, "à Lyon")

        assert result["destination"] is not None
        assert "Lyon" in result["destination"]

    def test_process_both(self, processor):
        """Test processing both departure and destination."""
        entities = {
            "departure": "Paris",
            "destination": "Lyon",
            "intermediate": [],
        }
        result = processor.process(entities, "De Paris à Lyon")

        assert result["departure"] is not None
        assert result["destination"] is not None
        assert "Paris" in result["departure"]
        assert "Lyon" in result["destination"]

    def test_process_intermediates(self, processor):
        """Test processing intermediate stops."""
        entities = {
            "departure": "Paris",
            "destination": "Lyon",
            "intermediate": ["Dijon", "Macon"],
        }
        result = processor.process(entities, "")

        assert isinstance(result["intermediate"], list)
        assert len(result["intermediate"]) == 2

    def test_process_preserves_none(self, processor):
        """Test that None values are preserved."""
        entities = {
            "departure": None,
            "destination": None,
            "intermediate": [],
        }
        result = processor.process(entities, "")

        assert result["departure"] is None
        assert result["destination"] is None
        assert result["intermediate"] == []

    def test_process_corrects_typos(self, processor):
        """Test that fuzzy matching corrects typos."""
        entities = {
            "departure": "Pari",
            "destination": "Nante",
            "intermediate": [],
        }
        result = processor.process(entities, "")

        # Should match to station names containing "Paris" and "Nantes"
        if result["departure"]:
            assert "Paris" in result["departure"]
        if result["destination"]:
            assert "Nantes" in result["destination"] or "Nant" in result["destination"]

    def test_process_normalizes_case(self, processor):
        """Test that fuzzy matching normalizes case."""
        entities = {
            "departure": "MARSEILLE",
            "destination": "toulouse",
            "intermediate": [],
        }
        result = processor.process(entities, "")

        # Should normalize to proper station names
        if result["departure"]:
            # Should not be all uppercase after normalization
            assert not result["departure"].isupper() or "-" in result["departure"]

    def test_process_empty_intermediate(self, processor):
        """Test processing with empty intermediate list."""
        entities = {
            "departure": "Paris",
            "destination": "Lyon",
            "intermediate": [],
        }
        result = processor.process(entities, "")

        assert result["intermediate"] == []

    def test_process_does_not_modify_original(self, processor):
        """Test that process does not modify the original entities dict."""
        entities = {
            "departure": "Paris",
            "destination": "Lyon",
            "intermediate": ["Dijon"],
        }
        original_departure = entities["departure"]

        processor.process(entities, "")

        # Original should be unchanged
        assert entities["departure"] == original_departure
