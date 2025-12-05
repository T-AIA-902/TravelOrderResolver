#!/bin/bash
# =============================================================================
# Travel Order Resolver - Run Model Benchmark
# =============================================================================

set -e

OUTPUT_DIR="evaluation/reports"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/benchmark_$TIMESTAMP.json"

echo "Running model benchmark..."
echo "Output will be saved to: $OUTPUT_FILE"

poetry run python evaluation/benchmark.py \
    --output "$OUTPUT_FILE" \
    --test-set datasets/processed/test.csv \
    "$@"

echo ""
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT_FILE"
