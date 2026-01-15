"""
Evaluate spaCy entity extractor on test dataset.

This script evaluates the spaCy-based entity extractor on the test sentences
and computes metrics (accuracy, precision, recall) for departure and destination
extraction.
"""

import sys
from pathlib import Path
import pandas as pd
import time

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.entity_extractor import SpacyEntityExtractor
from evaluation.metrics import calculate_entity_metrics, measure_latency


def normalize_location(location):
    """
    Normalize location name for comparison.

    Handles None values and strips whitespace.
    """
    if pd.isna(location) or location is None or location == "":
        return None
    return str(location).strip()


def evaluate_extractor(dataset_path: str):
    """
    Evaluate the entity extractor on a test dataset.

    Args:
        dataset_path: Path to the CSV file with test sentences

    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 70)
    print("spaCy Entity Extractor - Evaluation")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} sentences")

    # Initialize extractor
    print("\nInitializing spaCy extractor...")
    extractor = SpacyEntityExtractor()

    # Filter only TRIP sentences (we only evaluate entity extraction on travel queries)
    trip_df = df[df['intent'] == 'TRIP'].copy()
    print(f"\nEvaluating on {len(trip_df)} TRIP sentences")

    # Storage for predictions and ground truth
    predictions = []
    ground_truth = []
    errors = []
    latencies = []

    print("\n" + "-" * 70)
    print("Processing sentences...")
    print("-" * 70)

    for idx, row in trip_df.iterrows():
        sentence = row['sentence']
        true_departure = normalize_location(row['departure'])
        true_destination = normalize_location(row['destination'])

        # Measure latency for this sentence
        start_time = time.perf_counter()
        result = extractor.extract_entities(sentence)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

        pred_departure = normalize_location(result['departure'])
        pred_destination = normalize_location(result['destination'])

        # Store predictions and ground truth
        predictions.append((pred_departure, pred_destination))
        ground_truth.append((true_departure, true_destination))

        # Store errors for analysis
        if not (pred_departure == true_departure and pred_destination == true_destination):
            errors.append({
                'sentence': sentence,
                'true_departure': true_departure,
                'pred_departure': pred_departure,
                'true_destination': true_destination,
                'pred_destination': pred_destination,
                'departure_correct': pred_departure == true_departure,
                'destination_correct': pred_destination == true_destination
            })

    # Calculate comprehensive metrics
    metrics = calculate_entity_metrics(predictions, ground_truth)

    # Calculate latency statistics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    total = len(trip_df)
    no_prediction = sum(1 for p in predictions if p[0] is None and p[1] is None)
    no_prediction_rate = (no_prediction / total * 100) if total > 0 else 0

    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal sentences evaluated: {total}")

    print("\n" + "-" * 70)
    print("COMPREHENSIVE METRICS")
    print("-" * 70)
    print(f"\n{'Metric':<20} {'Departure':<15} {'Destination':<15} {'Both Correct':<15}")
    print("-" * 70)
    print(f"{'Precision':<20} {metrics['departure']['precision']:>13.2f}% {metrics['destination']['precision']:>14.2f}% {metrics['both']['precision']:>14.2f}%")
    print(f"{'Recall':<20} {metrics['departure']['recall']:>13.2f}% {metrics['destination']['recall']:>14.2f}% {metrics['both']['recall']:>14.2f}%")
    print(f"{'F1-Score':<20} {metrics['departure']['f1_score']:>13.2f}% {metrics['destination']['f1_score']:>14.2f}% {metrics['both']['f1_score']:>14.2f}%")
    print(f"{'Accuracy':<20} {metrics['departure']['accuracy']:>13.2f}% {metrics['destination']['accuracy']:>14.2f}% {metrics['both']['accuracy']:>14.2f}%")

    print("\n" + "-" * 70)
    print("LATENCY STATISTICS")
    print("-" * 70)
    print(f"Average latency:  {avg_latency:>6.2f} ms")
    print(f"Min latency:      {min_latency:>6.2f} ms")
    print(f"Max latency:      {max_latency:>6.2f} ms")
    print(f"Throughput:       {1000/avg_latency:>6.2f} sentences/sec")

    print("\n" + "-" * 70)
    print(f"No prediction (failed): {no_prediction} ({no_prediction_rate:.2f}%)")

    # Display error analysis
    if errors:
        print("\n" + "=" * 70)
        print(f"ERROR ANALYSIS ({len(errors)} errors)")
        print("=" * 70)

        for i, error in enumerate(errors[:10], 1):  # Show first 10 errors
            print(f"\n{i}. Sentence: {error['sentence']}")
            if not error['departure_correct']:
                print(f"   Departure:   TRUE={error['true_departure']:<15} PRED={error['pred_departure']}")
            if not error['destination_correct']:
                print(f"   Destination: TRUE={error['true_destination']:<15} PRED={error['pred_destination']}")

        if len(errors) > 10:
            print(f"\n... and {len(errors) - 10} more errors")

    print("\n" + "=" * 70)

    return {
        'total': total,
        'metrics': metrics,
        'latency': {
            'avg_ms': avg_latency,
            'min_ms': min_latency,
            'max_ms': max_latency
        },
        'no_prediction_rate': no_prediction_rate,
        'errors': errors
    }


def main():
    """Main evaluation function."""
    # Path to test dataset (using generated dataset for comprehensive evaluation)
    dataset_path = Path(__file__).parent.parent / "datasets" / "generated" / "test.csv"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Run evaluation
    results = evaluate_extractor(str(dataset_path))

    print("\nEvaluation complete!")
    print(f"Summary: {results['metrics']['both']['accuracy']:.1f}% of sentences fully correct")
    print(f"Average latency: {results['latency']['avg_ms']:.1f}ms")


if __name__ == "__main__":
    main()
