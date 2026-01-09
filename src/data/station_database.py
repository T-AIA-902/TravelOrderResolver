"""
Station database module for managing SNCF stations.

This module provides a unified database of French railway stations
with normalization, alias management, and search capabilities.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .loader import DataLoader


@dataclass
class Station:
    """
    Represents a railway station.

    Attributes:
        name: Official station name
        name_normalized: Normalized name (lowercase, no accents)
        short_code: Short code (e.g., "ABT" for Abancourt)
        uic_codes: List of UIC codes (unique identifiers)
        latitude: GPS latitude
        longitude: GPS longitude
        commune: Municipality name
        department: Department name
        is_passenger: Whether the station serves passengers
        is_freight: Whether the station serves freight
        line_codes: List of railway line codes serving this station
        aliases: Alternative names for the station
    """

    name: str
    name_normalized: str
    short_code: str = ""
    uic_codes: list[str] = field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    commune: str = ""
    department: str = ""
    is_passenger: bool = True
    is_freight: bool = False
    line_codes: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        """Hash based on UIC codes or name."""
        if self.uic_codes:
            return hash(tuple(sorted(self.uic_codes)))
        return hash(self.name_normalized)

    def __eq__(self, other: object) -> bool:
        """Equality based on UIC codes or name."""
        if not isinstance(other, Station):
            return False
        if self.uic_codes and other.uic_codes:
            return set(self.uic_codes) == set(other.uic_codes)
        return self.name_normalized == other.name_normalized


def normalize_name(name: str) -> str:
    """
    Normalize a station or city name for matching.

    Transformations:
    - Convert to lowercase
    - Remove accents (e -> e, etc.)
    - Replace hyphens and apostrophes with spaces
    - Remove extra whitespace
    - Remove common prefixes (gare de, gare du, etc.)

    Args:
        name: Original name string.

    Returns:
        Normalized name string.

    Examples:
        >>> normalize_name("Paris-Gare-de-Lyon")
        'paris gare de lyon'
        >>> normalize_name("Aeroport Charles de Gaulle")
        'aeroport charles de gaulle'
    """
    if not name:
        return ""

    # Convert to lowercase
    result = name.lower()

    # Remove accents using unicode normalization
    result = unicodedata.normalize("NFD", result)
    result = "".join(c for c in result if unicodedata.category(c) != "Mn")

    # Replace hyphens and apostrophes with spaces
    result = re.sub(r"[-'`]", " ", result)

    # Remove special characters except spaces
    result = re.sub(r"[^\w\s]", "", result)

    # Normalize whitespace
    result = " ".join(result.split())

    return result


def extract_city_name(station_name: str) -> str:
    """
    Extract the city name from a station name.

    Removes common station suffixes like "Gare", "TGV", "Ville", etc.

    Args:
        station_name: Full station name.

    Returns:
        Extracted city name.

    Examples:
        >>> extract_city_name("Paris Gare de Lyon")
        'Paris'
        >>> extract_city_name("Macon Loche TGV")
        'Macon Loche'
    """
    # Common suffixes to remove
    suffixes_to_remove = [
        r"\s+tgv$",
        r"\s+ville$",
        r"\s+gare\s+de\s+\w+$",
        r"\s+gare\s+du\s+\w+$",
        r"\s+st\s+\w+$",  # Keep for now, might be part of name
        r"\s+grand\s+cormier$",
        r"\s+country\s+club$",
        r"\s*\(intramuros\)$",
    ]

    result = station_name
    for pattern in suffixes_to_remove:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    return result.strip()


class StationDatabase:
    """
    Database for managing French railway stations.

    Provides unified access to station data from multiple SNCF sources,
    with normalization, alias management, and search capabilities.
    """

    def __init__(self, data_loader: DataLoader | None = None):
        """
        Initialize the station database.

        Args:
            data_loader: DataLoader instance. Creates a new one if not provided.
        """
        self.loader = data_loader or DataLoader()
        self._stations: dict[str, Station] = {}  # UIC code -> Station
        self._name_index: dict[str, list[str]] = {}  # normalized name -> UIC codes
        self._city_index: dict[str, list[str]] = {}  # city name -> UIC codes
        self._alias_index: dict[str, list[str]] = {}  # alias -> UIC codes
        self._loaded = False

    def load(self) -> None:
        """
        Load station data from SNCF files.

        Merges data from gares-de-voyageurs.json and liste-des-gares.json.
        """
        if self._loaded:
            return

        # Load passenger stations first (more detailed info)
        self._load_gares_voyageurs()

        # Enrich with liste-des-gares data
        self._load_liste_gares()

        # Build indexes
        self._build_indexes()

        # Generate aliases
        self._generate_aliases()

        self._loaded = True

    def _load_gares_voyageurs(self) -> None:
        """Load and parse gares-de-voyageurs.json."""
        try:
            data = self.loader.load_gares_voyageurs()
        except FileNotFoundError:
            return

        for entry in data:
            name = entry.get("nom", "")
            if not name:
                continue

            # Parse UIC codes (can be multiple separated by ;)
            uic_raw = entry.get("codes_uic", "")
            uic_codes = [c.strip() for c in uic_raw.split(";") if c.strip()]

            # Parse coordinates (may be None)
            geo = entry.get("position_geographique") or {}
            lat = geo.get("lat") if geo else None
            lon = geo.get("lon") if geo else None

            station = Station(
                name=name,
                name_normalized=normalize_name(name),
                short_code=entry.get("libellecourt", ""),
                uic_codes=uic_codes,
                latitude=lat,
                longitude=lon,
                is_passenger=True,
            )

            # Store by each UIC code
            for uic in uic_codes:
                self._stations[uic] = station

    def _load_liste_gares(self) -> None:
        """Load and enrich with liste-des-gares.json."""
        try:
            data = self.loader.load_liste_gares()
        except FileNotFoundError:
            return

        for entry in data:
            uic = entry.get("code_uic", "")
            if not uic:
                continue

            # Check if station already exists
            if uic in self._stations:
                # Enrich existing station
                station = self._stations[uic]
                if not station.commune:
                    station.commune = entry.get("commune", "")
                if not station.department:
                    station.department = entry.get("departemen", "")
                if entry.get("code_ligne"):
                    if entry["code_ligne"] not in station.line_codes:
                        station.line_codes.append(entry["code_ligne"])
                station.is_freight = entry.get("fret", "N") == "O"
            else:
                # Create new station (non-passenger or missing from first file)
                name = entry.get("libelle", "")
                if not name:
                    continue

                geo = entry.get("c_geo", {})
                lat = geo.get("lat") if geo else entry.get("y_wgs84")
                lon = geo.get("lon") if geo else entry.get("x_wgs84")

                station = Station(
                    name=name,
                    name_normalized=normalize_name(name),
                    uic_codes=[uic],
                    latitude=lat,
                    longitude=lon,
                    commune=entry.get("commune", ""),
                    department=entry.get("departemen", ""),
                    is_passenger=entry.get("voyageurs", "N") == "O",
                    is_freight=entry.get("fret", "N") == "O",
                    line_codes=[entry["code_ligne"]] if entry.get("code_ligne") else [],
                )
                self._stations[uic] = station

    def _build_indexes(self) -> None:
        """Build search indexes for station lookup."""
        for uic, station in self._stations.items():
            # Index by normalized name
            norm_name = station.name_normalized
            if norm_name not in self._name_index:
                self._name_index[norm_name] = []
            if uic not in self._name_index[norm_name]:
                self._name_index[norm_name].append(uic)

            # Index by city name
            city = normalize_name(extract_city_name(station.name))
            if city and city != norm_name:
                if city not in self._city_index:
                    self._city_index[city] = []
                if uic not in self._city_index[city]:
                    self._city_index[city].append(uic)

            # Index by commune
            if station.commune:
                commune_norm = normalize_name(station.commune)
                if commune_norm not in self._city_index:
                    self._city_index[commune_norm] = []
                if uic not in self._city_index[commune_norm]:
                    self._city_index[commune_norm].append(uic)

    def _generate_aliases(self) -> None:
        """Generate common aliases for stations."""
        # Common alias patterns
        alias_patterns: dict[str, list[str]] = {
            # Paris stations
            "paris gare de lyon": ["paris lyon", "gare de lyon"],
            "paris gare du nord": ["paris nord", "gare du nord"],
            "paris gare de l est": ["paris est", "gare de l est"],
            "paris gare saint lazare": ["paris st lazare", "saint lazare"],
            "paris gare montparnasse": ["paris montparnasse", "montparnasse"],
            "paris gare d austerlitz": ["paris austerlitz", "austerlitz"],
            # Airports
            "aeroport charles de gaulle 2 tgv": ["cdg", "roissy", "charles de gaulle"],
            "aeroport charles de gaulle 1": ["cdg 1", "roissy 1"],
            # Major cities with multiple stations
            "lyon part dieu": ["lyon", "part dieu"],
            "lyon perrache": ["perrache"],
            "marseille st charles": ["marseille", "saint charles"],
            "bordeaux st jean": ["bordeaux", "saint jean"],
            "lille flandres": ["lille"],
            "lille europe": ["lille europe tgv"],
        }

        for norm_name, aliases in alias_patterns.items():
            if norm_name in self._name_index:
                uic_codes = self._name_index[norm_name]
                for alias in aliases:
                    alias_norm = normalize_name(alias)
                    if alias_norm not in self._alias_index:
                        self._alias_index[alias_norm] = []
                    for uic in uic_codes:
                        if uic not in self._alias_index[alias_norm]:
                            self._alias_index[alias_norm].append(uic)
                        # Also add to station's aliases
                        if alias not in self._stations[uic].aliases:
                            self._stations[uic].aliases.append(alias)

    def get_station_by_uic(self, uic_code: str) -> Station | None:
        """
        Get a station by its UIC code.

        Args:
            uic_code: The UIC identifier.

        Returns:
            Station object or None if not found.
        """
        if not self._loaded:
            self.load()
        return self._stations.get(uic_code)

    def search_by_name(self, name: str, exact: bool = False) -> list[Station]:
        """
        Search for stations by name.

        Args:
            name: Station name to search for.
            exact: If True, only return exact matches.

        Returns:
            List of matching Station objects.
        """
        if not self._loaded:
            self.load()

        norm_name = normalize_name(name)
        results: list[Station] = []
        seen_uics: set[str] = set()

        # Exact match in name index
        if norm_name in self._name_index:
            for uic in self._name_index[norm_name]:
                if uic not in seen_uics:
                    results.append(self._stations[uic])
                    seen_uics.add(uic)

        if exact:
            return results

        # Check aliases
        if norm_name in self._alias_index:
            for uic in self._alias_index[norm_name]:
                if uic not in seen_uics:
                    results.append(self._stations[uic])
                    seen_uics.add(uic)

        # Check city index
        if norm_name in self._city_index:
            for uic in self._city_index[norm_name]:
                if uic not in seen_uics:
                    results.append(self._stations[uic])
                    seen_uics.add(uic)

        # Partial matching (name contains search term)
        if not results:
            for station_name, uic_codes in self._name_index.items():
                if norm_name in station_name or station_name in norm_name:
                    for uic in uic_codes:
                        if uic not in seen_uics:
                            results.append(self._stations[uic])
                            seen_uics.add(uic)

        return results

    def get_stations_by_city(self, city_name: str) -> list[Station]:
        """
        Get all stations in a city.

        Args:
            city_name: Name of the city.

        Returns:
            List of stations in the city.
        """
        if not self._loaded:
            self.load()

        norm_city = normalize_name(city_name)
        results: list[Station] = []
        seen_uics: set[str] = set()

        # Check city index
        if norm_city in self._city_index:
            for uic in self._city_index[norm_city]:
                if uic not in seen_uics:
                    results.append(self._stations[uic])
                    seen_uics.add(uic)

        # Also check name index for cities that are station names
        if norm_city in self._name_index:
            for uic in self._name_index[norm_city]:
                if uic not in seen_uics:
                    results.append(self._stations[uic])
                    seen_uics.add(uic)

        return results

    def get_all_stations(self, passenger_only: bool = True) -> list[Station]:
        """
        Get all stations in the database.

        Args:
            passenger_only: If True, only return passenger stations.

        Returns:
            List of all stations.
        """
        if not self._loaded:
            self.load()

        if passenger_only:
            return [s for s in self._stations.values() if s.is_passenger]
        return list(self._stations.values())

    def get_all_station_names(self, normalized: bool = False) -> list[str]:
        """
        Get all station names.

        Args:
            normalized: If True, return normalized names.

        Returns:
            List of station names.
        """
        if not self._loaded:
            self.load()

        if normalized:
            return list(self._name_index.keys())
        return [s.name for s in self._stations.values()]

    def get_all_city_names(self) -> list[str]:
        """
        Get all city names that have stations.

        Returns:
            List of city names.
        """
        if not self._loaded:
            self.load()

        return list(self._city_index.keys())

    def stats(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics.
        """
        if not self._loaded:
            self.load()

        passenger_stations = sum(1 for s in self._stations.values() if s.is_passenger)
        freight_stations = sum(1 for s in self._stations.values() if s.is_freight)

        return {
            "total_stations": len(self._stations),
            "passenger_stations": passenger_stations,
            "freight_stations": freight_stations,
            "unique_names": len(self._name_index),
            "cities": len(self._city_index),
            "aliases": len(self._alias_index),
        }
