"""
Pytest configuration and fixtures for TravelOrderResolver tests.
"""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_gares_voyageurs() -> list[dict]:
    """Sample data from gares-de-voyageurs.json."""
    return [
        {
            "nom": "Paris Gare de Lyon",
            "libellecourt": "PLY",
            "segment_drg": "A",
            "position_geographique": {"lon": 2.373462, "lat": 48.844922},
            "codeinsee": "75112",
            "codes_uic": "87686006",
        },
        {
            "nom": "Lyon Part Dieu",
            "libellecourt": "LPD",
            "segment_drg": "A",
            "position_geographique": {"lon": 4.859382, "lat": 45.760568},
            "codeinsee": "69003",
            "codes_uic": "87723197",
        },
        {
            "nom": "Marseille St Charles",
            "libellecourt": "MSC",
            "segment_drg": "A",
            "position_geographique": {"lon": 5.380410, "lat": 43.302765},
            "codeinsee": "13001",
            "codes_uic": "87751008",
        },
        {
            "nom": "Aeroport Charles de Gaulle 2 TGV",
            "libellecourt": "RYT",
            "segment_drg": "A;A",
            "position_geographique": {"lon": 2.570892, "lat": 49.003652},
            "codeinsee": "93073",
            "codes_uic": "87271494;87001479",
        },
    ]


@pytest.fixture
def sample_liste_gares() -> list[dict]:
    """Sample data from liste-des-gares.json."""
    return [
        {
            "code_uic": "87686006",
            "libelle": "Paris-Gare-de-Lyon",
            "fret": "N",
            "voyageurs": "O",
            "code_ligne": "830000",
            "commune": "PARIS",
            "departemen": "PARIS",
            "x_wgs84": 2.373462,
            "y_wgs84": 48.844922,
            "c_geo": {"lon": 2.373462, "lat": 48.844922},
        },
        {
            "code_uic": "87723197",
            "libelle": "Lyon-Part-Dieu",
            "fret": "N",
            "voyageurs": "O",
            "code_ligne": "830000",
            "commune": "LYON",
            "departemen": "RHONE",
            "x_wgs84": 4.859382,
            "y_wgs84": 45.760568,
            "c_geo": {"lon": 4.859382, "lat": 45.760568},
        },
        {
            "code_uic": "87999999",
            "libelle": "Gare Test Fret",
            "fret": "O",
            "voyageurs": "N",
            "code_ligne": "999000",
            "commune": "TEST",
            "departemen": "TEST",
            "x_wgs84": 0.0,
            "y_wgs84": 0.0,
            "c_geo": {"lon": 0.0, "lat": 0.0},
        },
    ]


@pytest.fixture
def temp_data_dir(
    sample_gares_voyageurs: list[dict], sample_liste_gares: list[dict]
) -> Path:
    """Create a temporary directory with sample SNCF data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Write sample gares-de-voyageurs.json
        with open(data_dir / "gares-de-voyageurs.json", "w", encoding="utf-8") as f:
            json.dump(sample_gares_voyageurs, f, ensure_ascii=False)

        # Write sample liste-des-gares.json
        with open(data_dir / "liste-des-gares.json", "w", encoding="utf-8") as f:
            json.dump(sample_liste_gares, f, ensure_ascii=False)

        yield data_dir
