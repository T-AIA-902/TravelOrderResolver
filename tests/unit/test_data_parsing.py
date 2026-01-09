"""
Unit tests for data parsing modules.

Tests the DataLoader and StationDatabase classes.
"""

from pathlib import Path

import pytest

from src.data import DataLoader, Station, StationDatabase, normalize_name


class TestNormalizeName:
    """Tests for the normalize_name function."""

    def test_lowercase(self) -> None:
        """Test lowercase conversion."""
        assert normalize_name("PARIS") == "paris"
        assert normalize_name("Paris") == "paris"

    def test_accent_removal(self) -> None:
        """Test accent removal."""
        assert normalize_name("Orleans") == "orleans"
        assert normalize_name("Beziers") == "beziers"
        assert normalize_name("Aeroport") == "aeroport"

    def test_hyphen_replacement(self) -> None:
        """Test hyphen replacement with space."""
        assert normalize_name("Paris-Lyon") == "paris lyon"
        assert normalize_name("Saint-Etienne") == "saint etienne"

    def test_apostrophe_replacement(self) -> None:
        """Test apostrophe replacement."""
        assert normalize_name("Gare de l'Est") == "gare de l est"

    def test_whitespace_normalization(self) -> None:
        """Test whitespace normalization."""
        assert normalize_name("Paris  Lyon") == "paris lyon"
        assert normalize_name("  Paris  ") == "paris"

    def test_empty_string(self) -> None:
        """Test empty string handling."""
        assert normalize_name("") == ""

    def test_complex_names(self) -> None:
        """Test complex station names."""
        assert normalize_name("Aeroport Charles de Gaulle 2 TGV") == "aeroport charles de gaulle 2 tgv"
        assert normalize_name("Paris Gare de Lyon") == "paris gare de lyon"


class TestStation:
    """Tests for the Station dataclass."""

    def test_station_creation(self) -> None:
        """Test basic station creation."""
        station = Station(
            name="Paris Gare de Lyon",
            name_normalized="paris gare de lyon",
            short_code="PLY",
            uic_codes=["87686006"],
        )
        assert station.name == "Paris Gare de Lyon"
        assert station.short_code == "PLY"
        assert "87686006" in station.uic_codes

    def test_station_equality_by_uic(self) -> None:
        """Test station equality based on UIC codes."""
        station1 = Station(
            name="Paris",
            name_normalized="paris",
            uic_codes=["12345"],
        )
        station2 = Station(
            name="Paris Gare",
            name_normalized="paris gare",
            uic_codes=["12345"],
        )
        assert station1 == station2

    def test_station_equality_by_name(self) -> None:
        """Test station equality based on normalized name when no UIC."""
        station1 = Station(name="Paris", name_normalized="paris")
        station2 = Station(name="PARIS", name_normalized="paris")
        assert station1 == station2

    def test_station_hash(self) -> None:
        """Test station hashing."""
        station = Station(
            name="Paris",
            name_normalized="paris",
            uic_codes=["12345"],
        )
        # Should be hashable
        station_set = {station}
        assert station in station_set


class TestDataLoader:
    """Tests for the DataLoader class."""

    def test_init_default_path(self) -> None:
        """Test default data directory path."""
        loader = DataLoader()
        assert loader.data_dir.name == "sncf"

    def test_init_custom_path(self, temp_data_dir: Path) -> None:
        """Test custom data directory path."""
        loader = DataLoader(temp_data_dir)
        assert loader.data_dir == temp_data_dir

    def test_file_exists(self, temp_data_dir: Path) -> None:
        """Test file existence check."""
        loader = DataLoader(temp_data_dir)
        assert loader.file_exists("gares-de-voyageurs.json")
        assert not loader.file_exists("nonexistent.json")

    def test_list_available_files(self, temp_data_dir: Path) -> None:
        """Test listing available files."""
        loader = DataLoader(temp_data_dir)
        files = loader.list_available_files()
        assert "gares-de-voyageurs.json" in files
        assert "liste-des-gares.json" in files

    def test_load_gares_voyageurs(self, temp_data_dir: Path) -> None:
        """Test loading gares-de-voyageurs.json."""
        loader = DataLoader(temp_data_dir)
        data = loader.load_gares_voyageurs()
        assert len(data) == 4
        assert data[0]["nom"] == "Paris Gare de Lyon"

    def test_load_liste_gares(self, temp_data_dir: Path) -> None:
        """Test loading liste-des-gares.json."""
        loader = DataLoader(temp_data_dir)
        data = loader.load_liste_gares()
        assert len(data) == 3
        assert data[0]["code_uic"] == "87686006"

    def test_load_nonexistent_file(self, temp_data_dir: Path) -> None:
        """Test loading nonexistent file raises error."""
        loader = DataLoader(temp_data_dir)
        with pytest.raises(FileNotFoundError):
            loader.load_lignes()


class TestStationDatabase:
    """Tests for the StationDatabase class."""

    def test_load_stations(self, temp_data_dir: Path) -> None:
        """Test loading stations from files."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)
        db.load()

        stats = db.stats()
        assert stats["total_stations"] >= 4

    def test_get_station_by_uic(self, temp_data_dir: Path) -> None:
        """Test getting station by UIC code."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)
        db.load()

        station = db.get_station_by_uic("87686006")
        assert station is not None
        assert "Paris" in station.name or "Lyon" in station.name

    def test_search_by_name_exact(self, temp_data_dir: Path) -> None:
        """Test exact name search."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)
        db.load()

        results = db.search_by_name("Paris Gare de Lyon", exact=True)
        assert len(results) >= 1

    def test_search_by_name_partial(self, temp_data_dir: Path) -> None:
        """Test partial name search."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)
        db.load()

        results = db.search_by_name("Lyon")
        assert len(results) >= 1

    def test_search_by_name_normalized(self, temp_data_dir: Path) -> None:
        """Test search with normalized input."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)
        db.load()

        # Search without accents should still work
        results = db.search_by_name("marseille")
        assert len(results) >= 1

    def test_get_all_stations(self, temp_data_dir: Path) -> None:
        """Test getting all stations."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)
        db.load()

        stations = db.get_all_stations(passenger_only=True)
        assert len(stations) >= 3  # At least 3 passenger stations

        all_stations = db.get_all_stations(passenger_only=False)
        assert len(all_stations) >= len(stations)

    def test_get_all_station_names(self, temp_data_dir: Path) -> None:
        """Test getting all station names."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)
        db.load()

        names = db.get_all_station_names(normalized=False)
        assert len(names) >= 3

        normalized_names = db.get_all_station_names(normalized=True)
        assert len(normalized_names) >= 1

    def test_lazy_loading(self, temp_data_dir: Path) -> None:
        """Test that database loads lazily."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)

        # Should not be loaded yet
        assert not db._loaded

        # Accessing data should trigger loading
        db.search_by_name("Paris")
        assert db._loaded

    def test_multiple_uic_codes(self, temp_data_dir: Path) -> None:
        """Test station with multiple UIC codes."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)
        db.load()

        # CDG has two UIC codes separated by ;
        station1 = db.get_station_by_uic("87271494")
        station2 = db.get_station_by_uic("87001479")

        # Both should return the same station
        assert station1 is not None
        assert station2 is not None
        assert station1 == station2

    def test_enrichment_from_liste_gares(self, temp_data_dir: Path) -> None:
        """Test that stations are enriched with liste-des-gares data."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)
        db.load()

        station = db.get_station_by_uic("87686006")
        assert station is not None
        # Should have commune from liste-des-gares
        assert station.commune == "PARIS"
        assert station.department == "PARIS"

    def test_stats(self, temp_data_dir: Path) -> None:
        """Test database statistics."""
        loader = DataLoader(temp_data_dir)
        db = StationDatabase(loader)
        db.load()

        stats = db.stats()
        assert "total_stations" in stats
        assert "passenger_stations" in stats
        assert "freight_stations" in stats
        assert stats["total_stations"] >= stats["passenger_stations"]
