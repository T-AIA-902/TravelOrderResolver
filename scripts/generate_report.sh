#!/bin/bash
# =============================================================================
# Travel Order Resolver - Generate Report
# =============================================================================

set -e

REPORT_DIR="docs/rapport"
mkdir -p "$REPORT_DIR"

echo "Generating project report..."

# Generate metrics summary
echo "Collecting metrics..."
poetry run python evaluation/metrics.py --summary > "$REPORT_DIR/metrics_summary.txt" 2>/dev/null || echo "Metrics not yet available"

# Generate confusion matrices
echo "Generating confusion matrices..."
poetry run python evaluation/confusion_matrix.py --output "$REPORT_DIR/confusion_matrices.png" 2>/dev/null || echo "Confusion matrices not yet available"

# TODO: Add LaTeX/PDF generation when report template is ready

echo ""
echo "Report generation complete!"
echo "Output directory: $REPORT_DIR"
