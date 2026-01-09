"""
Data loader module for SNCF datasets.

This module provides utilities to load and parse SNCF data files
including stations, lines, and schedules.
"""

import json
from pathlib import Path
from typing import Any

# Default paths for SNCF data files
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "datasets" / "raw" / "sncf"

GARES_VOYAGEURS_FILE = "gares-de-voyageurs.json"
LISTE_GARES_FILE = "liste-des-gares.json"
LIGNES_FILE = "lignes-par-type.json"
TGVMAX_FILE = "tgvmax.json"


class DataLoader:
    """
    Loader for SNCF data files.

    Handles loading and basic validation of JSON data files.
    """

    def __init__(self, data_dir: Path | str | None = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the directory containing SNCF data files.
                     Defaults to datasets/raw/sncf/
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    def _load_json(self, filename: str) -> list[dict[str, Any]]:
        """
        Load a JSON file from the data directory.

        Args:
            filename: Name of the JSON file to load.

        Returns:
            Parsed JSON data as a list of dictionaries.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def load_gares_voyageurs(self) -> list[dict[str, Any]]:
        """
        Load the passenger stations file (gares-de-voyageurs.json).

        Returns:
            List of station dictionaries with keys:
            - nom: Station name
            - libellecourt: Short code
            - segment_drg: DRG segment (A, B, C)
            - position_geographique: {lon, lat}
            - codeinsee: INSEE code
            - codes_uic: UIC code(s)
        """
        return self._load_json(GARES_VOYAGEURS_FILE)

    def load_liste_gares(self) -> list[dict[str, Any]]:
        """
        Load the complete stations list (liste-des-gares.json).

        Returns:
            List of station dictionaries with keys:
            - code_uic: UIC code
            - libelle: Station name
            - fret: Freight station (O/N)
            - voyageurs: Passenger station (O/N)
            - code_ligne: Line code
            - commune: Municipality name
            - departemen: Department name
            - x_wgs84, y_wgs84: GPS coordinates
        """
        return self._load_json(LISTE_GARES_FILE)

    def load_lignes(self) -> list[dict[str, Any]]:
        """
        Load the railway lines file (lignes-par-type.json).

        Returns:
            List of line dictionaries with keys:
            - type_ligne: Line type
            - code_ligne: Line code
            - lib_ligne: Line name
            - Geometry and coordinate information
        """
        return self._load_json(LIGNES_FILE)

    def load_tgvmax(self) -> list[dict[str, Any]]:
        """
        Load the TGVMax schedules file (tgvmax.json).

        Warning: This file is large (~103MB, 390k+ records).

        Returns:
            List of trip dictionaries with keys:
            - date: Trip date
            - train_no: Train number
            - origine, destination: Station names
            - origine_iata, destination_iata: IATA codes
            - heure_depart, heure_arrivee: Departure/arrival times
        """
        return self._load_json(TGVMAX_FILE)

    def get_data_dir(self) -> Path:
        """Return the current data directory path."""
        return self.data_dir

    def file_exists(self, filename: str) -> bool:
        """Check if a data file exists."""
        return (self.data_dir / filename).exists()

    def list_available_files(self) -> list[str]:
        """List all JSON files available in the data directory."""
        if not self.data_dir.exists():
            return []
        return [f.name for f in self.data_dir.glob("*.json")]
