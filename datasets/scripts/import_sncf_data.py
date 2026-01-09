#!/usr/bin/env python3
"""
SNCF Data Import Script.

This script loads and validates SNCF data files, providing statistics
and optionally exporting processed data for use in the NLP pipeline.

Usage:
    python import_sncf_data.py [--export] [--stats] [--sample N]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data import DataLoader, StationDatabase, normalize_name


def print_separator(char: str = "-", length: int = 60) -> None:
    """Print a separator line."""
    print(char * length)


def show_stats(db: StationDatabase) -> None:
    """Display database statistics."""
    stats = db.stats()

    print("\n=== STATION DATABASE STATISTICS ===\n")
    print(f"Total stations:     {stats['total_stations']:,}")
    print(f"Passenger stations: {stats['passenger_stations']:,}")
    print(f"Freight stations:   {stats['freight_stations']:,}")
    print(f"Unique names:       {stats['unique_names']:,}")
    print(f"Cities indexed:     {stats['cities']:,}")
    print(f"Aliases defined:    {stats['aliases']:,}")


def show_sample_stations(db: StationDatabase, count: int = 10) -> None:
    """Display sample stations."""
    print(f"\n=== SAMPLE STATIONS (first {count}) ===\n")

    stations = db.get_all_stations(passenger_only=True)[:count]

    for i, station in enumerate(stations, 1):
        print(f"{i}. {station.name}")
        print(f"   Code: {station.short_code or 'N/A'}")
        print(f"   UIC: {', '.join(station.uic_codes)}")
        if station.commune:
            print(f"   Commune: {station.commune}")
        if station.latitude and station.longitude:
            print(f"   Location: ({station.latitude:.4f}, {station.longitude:.4f})")
        print()


def test_search(db: StationDatabase) -> None:
    """Test search functionality with common queries."""
    print("\n=== SEARCH TESTS ===\n")

    test_queries = [
        "Paris",
        "Lyon",
        "Marseille",
        "paris gare de lyon",
        "CDG",
        "bordeaux",
        "lille",
        "Rennes",
        "Nantes",
        "Strasbourg",
    ]

    for query in test_queries:
        results = db.search_by_name(query)
        print(f"'{query}' -> {len(results)} result(s)")
        if results:
            # Show first 3 results
            for station in results[:3]:
                print(f"   - {station.name}")
            if len(results) > 3:
                print(f"   ... and {len(results) - 3} more")
        print()


def export_station_names(db: StationDatabase, output_path: Path) -> None:
    """Export station names to a text file for NLP dataset generation."""
    print(f"\nExporting station names to {output_path}...")

    stations = db.get_all_stations(passenger_only=True)

    # Collect unique names and their normalized versions
    names_data = []
    for station in stations:
        names_data.append(
            {
                "name": station.name,
                "normalized": station.name_normalized,
                "short_code": station.short_code,
                "commune": station.commune,
                "aliases": station.aliases,
            }
        )

    # Write to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(names_data, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(names_data)} stations.")


def export_city_mapping(db: StationDatabase, output_path: Path) -> None:
    """Export city to station mapping."""
    print(f"\nExporting city mapping to {output_path}...")

    cities = db.get_all_city_names()
    mapping = {}

    for city in cities:
        stations = db.get_stations_by_city(city)
        mapping[city] = [s.name for s in stations]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(mapping)} city mappings.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import and validate SNCF station data."
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show database statistics"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Show N sample stations",
    )
    parser.add_argument(
        "--search", action="store_true", help="Run search tests"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export processed data to datasets/processed/",
    )
    parser.add_argument(
        "--query",
        type=str,
        metavar="NAME",
        help="Search for a specific station",
    )

    args = parser.parse_args()

    # Default to showing stats if no options provided
    if not any([args.stats, args.sample, args.search, args.export, args.query]):
        args.stats = True
        args.sample = 5
        args.search = True

    print("Loading SNCF station data...")
    print_separator()

    # Initialize database
    loader = DataLoader()
    db = StationDatabase(loader)

    # Check data files exist
    available_files = loader.list_available_files()
    print(f"Available data files: {', '.join(available_files)}")

    # Load data
    try:
        db.load()
        print("Data loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure SNCF data files are in datasets/raw/sncf/")
        sys.exit(1)

    print_separator()

    # Execute requested operations
    if args.stats:
        show_stats(db)

    if args.sample > 0:
        show_sample_stations(db, args.sample)

    if args.search:
        test_search(db)

    if args.query:
        print(f"\n=== SEARCH: '{args.query}' ===\n")
        results = db.search_by_name(args.query)
        if results:
            for station in results:
                print(f"- {station.name}")
                print(f"  UIC: {', '.join(station.uic_codes)}")
                print(f"  Commune: {station.commune}")
                print(f"  Normalized: {station.name_normalized}")
                print()
        else:
            print("No results found.")

    if args.export:
        processed_dir = Path(__file__).parent.parent / "processed"

        export_station_names(db, processed_dir / "stations.json")
        export_city_mapping(db, processed_dir / "city_mapping.json")

        print("\nExport complete!")


if __name__ == "__main__":
    main()
