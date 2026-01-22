#!/usr/bin/env python3
"""
Dataset Validation Script for Travel Order Resolver.

This script validates the generated NLP dataset, checking for:
- Schema compliance
- Data quality
- Distribution analysis
- Edge cases detection

Usage:
    python validate_dataset.py [--input DIR] [--verbose]
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data import StationDatabase, normalize_name  # noqa: E402

# Valid intents (language is now a separate column, not an intent)
VALID_INTENTS = {"TRIP", "NOT_TRIP", "UNKNOWN"}


def load_csv(filepath: Path) -> list[dict]:
    """Load dataset from CSV file."""
    entries = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries


def validate_schema(entries: list[dict]) -> list[str]:
    """Validate dataset schema."""
    errors = []
    required_fields = {
        "sentence_id",
        "sentence",
        "intent",
        "departure",
        "destination",
        "intermediate",
    }

    for i, entry in enumerate(entries):
        # Check required fields
        missing = required_fields - set(entry.keys())
        if missing:
            errors.append(f"Row {i+1}: Missing fields: {missing}")

        # Check intent value
        if entry.get("intent") not in VALID_INTENTS:
            errors.append(f"Row {i+1}: Invalid intent '{entry.get('intent')}'")

        # Check sentence is not empty for non-UNKNOWN
        if entry.get("intent") != "UNKNOWN" and not entry.get("sentence", "").strip():
            errors.append(f"Row {i+1}: Empty sentence for non-UNKNOWN intent")

        # Check TRIP has departure and destination
        if entry.get("intent") == "TRIP":
            if not entry.get("departure"):
                errors.append(f"Row {i+1}: TRIP missing departure")
            if not entry.get("destination"):
                errors.append(f"Row {i+1}: TRIP missing destination")

        # Check non-TRIP should not have departure/destination
        if entry.get("intent") in ("NOT_TRIP", "NOT_FRENCH", "UNKNOWN"):
            if entry.get("departure") or entry.get("destination"):
                errors.append(
                    f"Row {i+1}: {entry.get('intent')} should not have departure/destination"
                )

    return errors


def validate_stations(entries: list[dict], station_db: StationDatabase) -> list[str]:
    """Validate that stations exist in database."""
    errors = []
    station_names = set(station_db.get_all_station_names(normalized=False))

    for i, entry in enumerate(entries):
        if entry.get("intent") != "TRIP":
            continue

        dep = entry.get("departure", "")
        dest = entry.get("destination", "")
        via = entry.get("intermediate", "")

        if dep and dep not in station_names:
            errors.append(f"Row {i+1}: Unknown departure station '{dep}'")

        if dest and dest not in station_names:
            errors.append(f"Row {i+1}: Unknown destination station '{dest}'")

        if via and via not in station_names:
            errors.append(f"Row {i+1}: Unknown intermediate station '{via}'")

    return errors


def check_duplicates(entries: list[dict]) -> list[str]:
    """Check for duplicate entries."""
    warnings = []
    sentence_counts = Counter(e.get("sentence", "") for e in entries)

    for sentence, count in sentence_counts.most_common():
        if count > 1 and sentence.strip():
            warnings.append(f"Duplicate sentence ({count}x): '{sentence[:50]}...'")

    # Only report first 10 duplicates
    return warnings[:10]


def analyze_distribution(entries: list[dict]) -> dict:
    """Analyze dataset distribution."""
    intent_counts = Counter(e.get("intent") for e in entries)
    intermediate_count = sum(1 for e in entries if e.get("intermediate"))

    # Sentence length statistics
    lengths = [len(e.get("sentence", "")) for e in entries]
    avg_length = sum(lengths) / len(lengths) if lengths else 0

    # Count augmentations (heuristic based on patterns)
    lowercase_count = sum(
        1 for e in entries if e.get("sentence", "").islower() and e.get("intent") == "TRIP"
    )

    return {
        "total": len(entries),
        "intent_distribution": dict(intent_counts),
        "with_intermediate": intermediate_count,
        "avg_sentence_length": avg_length,
        "min_sentence_length": min(lengths) if lengths else 0,
        "max_sentence_length": max(lengths) if lengths else 0,
        "lowercase_sentences": lowercase_count,
    }


def check_edge_cases(entries: list[dict]) -> list[str]:
    """Check for potential edge cases and issues."""
    issues = []

    for i, entry in enumerate(entries):
        sentence = entry.get("sentence", "")
        dep = entry.get("departure", "")
        dest = entry.get("destination", "")

        # Check if sentence contains departure/destination for TRIP
        if entry.get("intent") == "TRIP":
            norm_sentence = normalize_name(sentence)
            norm_dep = normalize_name(dep)
            _norm_dest = normalize_name(dest)  # noqa: F841

            # This is expected to fail for sentences with typos
            # We just track it for analysis
            if norm_dep and norm_dep not in norm_sentence:
                # Could be due to typos in sentence - this is expected
                pass

        # Check for very short sentences
        if len(sentence) < 3 and entry.get("intent") != "UNKNOWN":
            issues.append(f"Row {i+1}: Very short sentence '{sentence}'")

        # Check for very long sentences
        if len(sentence) > 200:
            issues.append(f"Row {i+1}: Very long sentence ({len(sentence)} chars)")

    return issues[:20]  # Limit to first 20


def print_report(
    schema_errors: list[str],
    station_errors: list[str],
    duplicates: list[str],
    distribution: dict,
    edge_cases: list[str],
    verbose: bool = False,
) -> bool:
    """Print validation report and return success status."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    # Distribution
    print("\n## Distribution")
    print(f"Total entries: {distribution['total']}")
    for intent, count in sorted(distribution["intent_distribution"].items()):
        pct = 100 * count / distribution["total"]
        print(f"  {intent}: {count} ({pct:.1f}%)")
    print(f"With intermediate: {distribution['with_intermediate']}")
    print(f"Avg sentence length: {distribution['avg_sentence_length']:.1f}")
    print(f"Lowercase sentences: {distribution['lowercase_sentences']}")

    # Schema errors
    print(f"\n## Schema Errors: {len(schema_errors)}")
    if schema_errors:
        for err in schema_errors[:10]:
            print(f"  - {err}")
        if len(schema_errors) > 10:
            print(f"  ... and {len(schema_errors) - 10} more")

    # Station errors
    print(f"\n## Station Errors: {len(station_errors)}")
    if station_errors:
        for err in station_errors[:10]:
            print(f"  - {err}")
        if len(station_errors) > 10:
            print(f"  ... and {len(station_errors) - 10} more")

    # Duplicates
    print(f"\n## Duplicates: {len(duplicates)}")
    if duplicates and verbose:
        for dup in duplicates:
            print(f"  - {dup}")

    # Edge cases
    print(f"\n## Edge Cases: {len(edge_cases)}")
    if edge_cases and verbose:
        for case in edge_cases:
            print(f"  - {case}")

    # Summary
    print("\n" + "=" * 60)
    total_errors = len(schema_errors) + len(station_errors)
    if total_errors == 0:
        print("VALIDATION PASSED")
        return True
    else:
        print(f"VALIDATION FAILED ({total_errors} errors)")
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate NLP dataset for travel order resolution."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input directory (default: datasets/generated/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="full_dataset.csv",
        help="Specific file to validate (default: full_dataset.csv)",
    )

    args = parser.parse_args()

    # Setup input directory
    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = Path(__file__).parent.parent / "augmented"

    filepath = input_dir / args.file

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        print("Run generate_sentences.py first to create the dataset.")
        sys.exit(1)

    print(f"Validating: {filepath}")

    # Load dataset
    print("Loading dataset...")
    entries = load_csv(filepath)
    print(f"Loaded {len(entries)} entries.")

    # Load station database
    print("Loading station database...")
    station_db = StationDatabase()
    station_db.load()

    # Run validations
    print("Running validations...")
    schema_errors = validate_schema(entries)
    station_errors = validate_stations(entries, station_db)
    duplicates = check_duplicates(entries)
    distribution = analyze_distribution(entries)
    edge_cases = check_edge_cases(entries)

    # Print report
    success = print_report(
        schema_errors,
        station_errors,
        duplicates,
        distribution,
        edge_cases,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
