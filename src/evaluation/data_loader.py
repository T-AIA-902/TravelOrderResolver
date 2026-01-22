"""
Data loading utilities for evaluation.

Handles loading datasets from JSON/CSV and normalizing data for comparison.
"""

import csv
import json
from pathlib import Path
from typing import Any


def load_dataset(filepath: str) -> list[dict[str, Any]]:
    """
    Load dataset from JSON or CSV file.

    Returns normalized format with keys: sentence, intent, language, departure, destination

    Args:
        filepath: Path to JSON or CSV file

    Returns:
        List of dictionaries with normalized keys
    """
    path = Path(filepath)

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)
            for sample in data:
                if "language" in sample:
                    sample["language"] = map_language_label(sample["language"])
            return data

    elif path.suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = []
            for row in reader:
                sample = {
                    "sentence": row.get("text", row.get("sentence", "")),
                    "intent": map_label_to_intent(row.get("label", row.get("intent", ""))),
                    "language": map_language_label(row.get("language", "UNKNOWN")),
                    "departure": row.get("departure", ""),
                    "destination": row.get("destination", ""),
                    "intermediate": row.get("intermediate", ""),
                }
                data.append(sample)
        return data

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def map_label_to_intent(label: str) -> str:
    """Map CSV labels to standard intent format."""
    label = label.upper().strip()
    if label == "VALID":
        return "TRIP"
    elif label == "INVALID":
        return "NOT_TRIP"
    return label


def map_language_label(label: str) -> str:
    """
    Map dataset language labels to 3-class system.

    FR/EN focused study - ES/DE/IT mapped to UNKNOWN.

    Args:
        label: Original language label from dataset

    Returns:
        One of "FRENCH", "ENGLISH", "UNKNOWN"
    """
    label = label.upper().strip()
    if label == "FRENCH":
        return "FRENCH"
    elif label == "ENGLISH":
        return "ENGLISH"
    else:
        return "UNKNOWN"


def normalize_location(location: str | None) -> str | None:
    """Normalize location for comparison."""
    if location is None or location == "":
        return None
    return str(location).strip()


def normalize_fuzzy_station(station: str | None) -> str | None:
    """Extract city name from full SNCF station name."""
    if station is None or station == "":
        return None
    station = str(station).strip()
    if "-" in station:
        return station.split("-")[0]
    return station


def is_misspelled_sample(sentence: str, departure: str, destination: str) -> bool:
    """Detect if sentence contains misspelled station names."""
    sentence_lower = sentence.lower()
    if departure and departure.lower() not in sentence_lower:
        return True
    if destination and destination.lower() not in sentence_lower:
        return True
    return False


def is_lowercase_sample(sentence: str) -> bool:
    """Detect if sentence starts with lowercase."""
    if not sentence:
        return False
    for char in sentence:
        if char.isalpha():
            return char.islower()
    return False
