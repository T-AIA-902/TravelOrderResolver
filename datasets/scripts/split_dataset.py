#!/usr/bin/env python3
"""
Split dataset into train/val/test sets (70/15/15).

This script splits dataset_train_sncf.csv into:
- train.csv (70%)
- val.csv (15%)
- test.csv (15%)

Usage:
    python datasets/scripts/split_dataset.py
    python datasets/scripts/split_dataset.py --seed 42
"""

import argparse
import csv
import random
from pathlib import Path


def split_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split a CSV dataset into train/val/test sets.

    Args:
        input_path: Path to input CSV file
        output_dir: Directory for output files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"

    print(f"Loading dataset from: {input_path}")

    # Read all data
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data = list(reader)

    total = len(data)
    print(f"Total samples: {total}")

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(data)

    # Calculate split indices
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"\nSplit (seed={seed}):")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/total*100:.1f}%)")
    print(f"  Val:   {len(val_data)} samples ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} samples ({len(test_data)/total*100:.1f}%)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write splits
    for name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        filepath = output_path / f"{name}.csv"
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_data)
        print(f"  Written: {filepath}")

    # Print label distribution for test set
    print("\nTest set label distribution:")
    label_counts = {}
    for sample in test_data:
        label = sample.get("label", "UNKNOWN")
        label_counts[label] = label_counts.get(label, 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(test_data)*100:.1f}%)")

    return {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
    }


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV file (default: datasets/processed/dataset_train_sncf.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: datasets/splits/)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio")

    args = parser.parse_args()

    # Default paths
    script_dir = Path(__file__).parent
    datasets_dir = script_dir.parent

    input_path = args.input or (datasets_dir / "processed" / "dataset_train_sncf.csv")
    output_dir = args.output_dir or (datasets_dir / "splits")

    # Run split
    split_dataset(
        input_path=str(input_path),
        output_dir=str(output_dir),
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
