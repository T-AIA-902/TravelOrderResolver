"""
Data management module for SNCF datasets.

This module provides utilities for loading, parsing, and managing
railway station and schedule data from SNCF open data sources.
"""

from .loader import DataLoader
from .station_database import Station, StationDatabase, normalize_name

__all__ = [
    "DataLoader",
    "Station",
    "StationDatabase",
    "normalize_name",
]
