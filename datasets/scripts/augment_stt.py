#!/usr/bin/env python3
"""
STT Augmentation Script for Travel Order Resolver.

Applies realistic Whisper speech-to-text errors to a clean base dataset.
This allows for controlled experiments comparing model performance
on clean vs augmented data.

Usage:
    python augment_stt.py [--input DIR] [--output DIR] [--seed S]

Example:
    python augment_stt.py --input datasets/base/ --output datasets/augmented/
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.stt_augmentation import ERROR_PROFILES, STTAugmenter, STTErrorConfig  # noqa: E402


def pick_intensity() -> str:
    """Pick STT error intensity based on distribution."""
    r = random.random()
    if r < 0.20:
        return "clean"
    elif r < 0.60:
        return "light"
    elif r < 0.90:
        return "moderate"
    else:
        return "heavy"


def create_trip_augmenters(seed: int | None = None) -> dict[str, STTAugmenter]:
    """Create TRIP-safe augmenters (no truncation to preserve ground truth)."""
    augmenters: dict[str, STTAugmenter] = {}
    for intensity in ["clean", "light", "moderate", "heavy"]:
        config = ERROR_PROFILES[intensity]
        trip_config = STTErrorConfig(
            filler_words=config.filler_words,
            false_starts=config.false_starts,
            repetitions=config.repetitions,
            incomplete=0.0,  # DISABLED for TRIP entries
            phonetic_confusion=config.phonetic_confusion,
            station_misspelling=config.station_misspelling,
            punctuation_errors=config.punctuation_errors,
            capitalization_errors=config.capitalization_errors,
            number_format=config.number_format,
            code_switching=config.code_switching,
            hallucinations=config.hallucinations,
            noise_artifacts=config.noise_artifacts,
            accent_variations=config.accent_variations,
            homophones=config.homophones,
        )
        augmenters[intensity] = STTAugmenter(config=trip_config, seed=seed)
    return augmenters


def create_general_augmenters(seed: int | None = None) -> dict[str, STTAugmenter]:
    """Create general augmenters (with all error types)."""
    augmenters: dict[str, STTAugmenter] = {}
    for intensity in ["clean", "light", "moderate", "heavy"]:
        config = ERROR_PROFILES[intensity]
        augmenters[intensity] = STTAugmenter(config=config, seed=seed)
    return augmenters


def augment_row(
    row: dict,
    trip_augmenters: dict[str, STTAugmenter],
    general_augmenters: dict[str, STTAugmenter],
) -> dict:
    """Augment a single row with STT errors."""
    intent = row.get("intent", "")
    sentence = row.get("sentence", "")

    # Pick augmenter based on intent
    if intent == "TRIP":
        # Use TRIP-safe augmenters (no truncation)
        intensity = pick_intensity()
        augmenter = trip_augmenters[intensity]

        # Get station to potentially corrupt
        station = row.get("departure") or row.get("destination") or ""
        apply_to_station = station if station and random.random() < 0.3 else None

        augmented = augmenter.augment(sentence, apply_to_station=apply_to_station)
    elif intent == "NOT_TRIP":
        # Use general augmenters with light augmentation
        if random.random() < 0.3:
            intensity = random.choice(["light", "moderate"])
            augmented = general_augmenters[intensity].augment(sentence)
        else:
            augmented = sentence
    else:
        # UNKNOWN - keep as is (already noisy)
        augmented = sentence

    # Update row with augmented sentence
    result = row.copy()
    result["sentence"] = augmented

    # Update sentence_id prefix from BASE to STT
    if result.get("sentence_id", "").startswith("BASE"):
        result["sentence_id"] = "STT" + result["sentence_id"][4:]

    return result


def augment_csv(
    input_path: Path,
    output_path: Path,
    trip_augmenters: dict[str, STTAugmenter],
    general_augmenters: dict[str, STTAugmenter],
) -> int:
    """Augment a CSV file and write to output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []

        with open(output_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            count = 0
            for row in reader:
                augmented_row = augment_row(row, trip_augmenters, general_augmenters)
                writer.writerow(augmented_row)
                count += 1

    return count


def augment_json(
    input_path: Path,
    output_path: Path,
    trip_augmenters: dict[str, STTAugmenter],
    general_augmenters: dict[str, STTAugmenter],
) -> int:
    """Augment a JSON file and write to output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    augmented_data = [augment_row(row, trip_augmenters, general_augmenters) for row in data]

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(augmented_data, outfile, ensure_ascii=False, indent=2)

    return len(augmented_data)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Apply STT augmentation to a clean base dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input directory (default: datasets/base/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: datasets/augmented/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="File format to process (default: both)",
    )

    args = parser.parse_args()

    # Setup directories
    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = Path(__file__).parent.parent / "base"

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "augmented"

    # Initialize random seed
    if args.seed is not None:
        random.seed(args.seed)

    # Create augmenters
    print("Initializing STT augmenters...")
    trip_augmenters = create_trip_augmenters(args.seed)
    general_augmenters = create_general_augmenters(args.seed)

    # Process CSV files
    if args.format in ("csv", "both"):
        csv_files = ["train.csv", "val.csv", "test.csv", "base_dataset_100k.csv"]
        for filename in csv_files:
            input_path = input_dir / filename
            if input_path.exists():
                # Map base_dataset_100k.csv to stt_dataset_100k.csv
                output_filename = filename.replace("base_dataset", "stt_dataset")
                output_path = output_dir / output_filename
                print(f"Augmenting {filename}...")
                count = augment_csv(input_path, output_path, trip_augmenters, general_augmenters)
                print(f"  Wrote {count} entries to {output_path}")

    # Process JSON files
    if args.format in ("json", "both"):
        json_files = ["train.json", "val.json", "test.json", "base_dataset_100k.json"]
        for filename in json_files:
            input_path = input_dir / filename
            if input_path.exists():
                # Map base_dataset_100k.json to stt_dataset_100k.json
                output_filename = filename.replace("base_dataset", "stt_dataset")
                output_path = output_dir / output_filename
                print(f"Augmenting {filename}...")
                count = augment_json(input_path, output_path, trip_augmenters, general_augmenters)
                print(f"  Wrote {count} entries to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
