#!/usr/bin/env bash
# run_all.sh - Reproduce all analyses and tests end-to-end
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Python interpreter (virtual environment if available, else system python)
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "=========================================================="
echo " Cancer Transcriptomics Machine Learning Pipeline Runner "
echo " Using Python: $($PYTHON --version) ($PYTHON)"
echo "=========================================================="

echo ""
echo "[Step 1/5] Running automated test suite with pytest..."
$PYTHON -m pytest -q

echo ""
echo "[Step 2/5] Checking / regenerating synthetic dataset..."
$PYTHON src/make_synthetic_dataset.py --out data/synthetic

echo ""
echo "[Step 3/5] Training models and evaluating cross-validation & test performance..."
$PYTHON src/train.py --data data/synthetic/synth_expression_log2cpm.csv --out results

echo ""
echo "[Step 4/5] Validating biomarker recovery against synthetic ground truth..."
$PYTHON src/validate_biomarker_recovery.py --found results/top_biomarkers.csv --truth data/synthetic/ground_truth_biomarkers.csv

echo ""
echo "[Step 5/6] Performing statistical differential expression (t-test + FDR)..."
$PYTHON src/differential_expression.py --data data/synthetic/synth_expression_log2cpm.csv --out results

echo ""
echo "[Step 6/6] Exporting web application model parameters for GitHub Live Sites..."
$PYTHON src/export_web_model.py
cp docs/model_data.js model_data.js

echo ""
echo "=========================================================="
echo " Pipeline completed successfully!"
echo " All plots, metrics, web UI, and biomarker lists ready!"
echo "=========================================================="
