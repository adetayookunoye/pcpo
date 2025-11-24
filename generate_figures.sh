#!/bin/bash
# Script to generate all publication figures

cd "$(dirname "$0")/../.."

echo "════════════════════════════════════════════════════════════════════════════"
echo "GENERATING PUBLICATION FIGURES FOR PCPO"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if results exist
if [ ! -f "results/comparison_metrics_seed0.json" ]; then
    echo "ERROR: Results not found. Run 'make compare' first."
    exit 1
fi

# Create output directory
mkdir -p results/figures

# Run the figure generation script
echo "Running figure generation..."
echo ""

python -m src.analysis.generate_publication_figures \
    --config config.yaml \
    --results-dir results \
    --outdir results/figures

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "✅ FIGURES GENERATED SUCCESSFULLY"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Generated figures:"
    ls -lh results/figures/fig*.png | awk '{print "  " $9}'
    echo ""
    echo "To view figures: open results/figures/ or use 'display fig1_*.png'"
    echo ""
else
    echo "❌ Error generating figures (exit code: $exit_code)"
    exit $exit_code
fi
