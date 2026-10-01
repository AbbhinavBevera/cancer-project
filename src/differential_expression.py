"""Univariate differential expression analysis.

Compares gene expression between tumour and normal samples using Welch's t-test
and Benjamini-Hochberg false discovery rate (FDR) correction.
This provides a classical statistical baseline to compare against the
machine-learning stability-selection biomarker rankings.
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats

from preprocessing import basic_clean, load_expression


def compute_differential_expression(X, y):
    """Compute Welch's t-test, log2 fold change, and Benjamini-Hochberg FDR."""
    tumour_mask = (y == 1)
    normal_mask = (y == 0)

    X_tumour = X[tumour_mask]
    X_normal = X[normal_mask]

    mean_tumour = X_tumour.mean(axis=0)
    mean_normal = X_normal.mean(axis=0)
    
    # Calculate effect size (difference in means for log-scale data, or log2FC)
    log2_fc = mean_tumour - mean_normal

    # Welch's two-sample t-test (unequal variances assumed)
    t_stat, p_vals = stats.ttest_ind(X_tumour, X_normal, axis=0, equal_var=False)
    
    # Handle NaNs if any zero variance features exist
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    t_stat = np.nan_to_num(t_stat, nan=0.0)

    # Benjamini-Hochberg FDR adjustment
    n_genes = len(p_vals)
    sorted_order = np.argsort(p_vals)
    sorted_pvals = p_vals[sorted_order]
    
    # FDR = p * n / rank
    ranks = np.arange(1, n_genes + 1)
    fdr = sorted_pvals * n_genes / ranks
    # Enforce monotonicity
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr = np.clip(fdr, 0, 1.0)

    fdr_adjusted = np.empty(n_genes)
    fdr_adjusted[sorted_order] = fdr

    de_table = pd.DataFrame({
        "gene": X.columns,
        "mean_tumour": mean_tumour.values,
        "mean_normal": mean_normal.values,
        "log2_fold_change": log2_fc.values,
        "direction": np.where(log2_fc > 0, "higher in tumour", "lower in tumour"),
        "t_statistic": t_stat,
        "p_value": p_vals,
        "fdr_bh": fdr_adjusted,
    }).set_index("gene")

    return de_table.sort_values(by=["fdr_bh", "p_value"])


def main():
    p = argparse.ArgumentParser(description="Univariate differential expression analysis")
    p.add_argument("--data", default="data/expression.csv")
    p.add_argument("--label-col", default="label")
    p.add_argument("--out", default="results")
    p.add_argument("--log-transform", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.data == "data/expression.csv" and not os.path.exists(args.data):
        synth_path = "data/synthetic/synth_expression_log2cpm.csv"
        if os.path.exists(synth_path):
            print(f"[Notice] '{args.data}' not found. Defaulting to demonstration data: '{synth_path}'.")
            args.data = synth_path

    X, y = load_expression(args.data, args.label_col)
    X = basic_clean(X, log_transform=args.log_transform)

    print(f"Running differential expression across {X.shape[1]} genes on {X.shape[0]} samples...")
    de_table = compute_differential_expression(X, y)

    out_file = os.path.join(args.out, "differential_expression.csv")
    de_table.round(6).to_csv(out_file)
    print(f"Saved differential expression results to {out_file}")

    sig_genes = (de_table["fdr_bh"] < 0.05).sum()
    print(f"Significant genes at FDR < 0.05: {sig_genes}")
    print("\nTop 10 differentially expressed genes:")
    print(de_table[["log2_fold_change", "direction", "p_value", "fdr_bh"]].head(10))


if __name__ == "__main__":
    main()
