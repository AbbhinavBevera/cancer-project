"""Generate a reproducible SYNTHETIC RNA-seq-like dataset (tumour vs normal).

THIS IS NOT REAL PATIENT DATA. It exists so that code, tests and AI agents
can be checked against a case where the right answer is known: the file
ground_truth_biomarkers.csv lists exactly which genes were made different.

Realistic features built in on purpose:
  - raw counts drawn from a negative binomial (overdispersed, like RNA-seq)
  - different sequencing depth per sample (library size)
  - a batch effect on ~15% of genes (a nuisance signal, unrelated to disease)
  - some true biomarkers are lowly expressed, so they are harder to find

Usage:
    python src/make_synthetic_dataset.py --out data/synthetic
"""
import argparse
import os

import numpy as np
import pandas as pd


def generate(n_samples=240, n_genes=1500, n_de=40, n_batches=3,
             tumour_frac=0.55, seed=7):
    rng = np.random.default_rng(seed)
    genes = [f"SYN_{i + 1:04d}" for i in range(n_genes)]
    samples = [f"SYN_S{i + 1:03d}" for i in range(n_samples)]

    n_t = int(round(n_samples * tumour_frac))
    y = np.array([1] * n_t + [0] * (n_samples - n_t))
    rng.shuffle(y)
    batch = rng.integers(0, n_batches, size=n_samples)

    base = np.exp(rng.normal(np.log(60), 1.4, size=n_genes))   # mean counts
    disp = rng.uniform(0.2, 0.8, size=n_genes)                # NB dispersion

    # true biomarkers: 60% up in tumour, 40% down
    de_idx = rng.choice(n_genes, size=n_de, replace=False)
    lfc = np.zeros(n_genes)
    mags = rng.uniform(0.25, 0.9, size=n_de)
    n_up = int(n_de * 0.6)
    signs = np.array([1] * n_up + [-1] * (n_de - n_up))
    rng.shuffle(signs)
    lfc[de_idx] = mags * signs

    # batch effect (log2 scale) on 15% of genes
    batch_fx = np.zeros((n_batches, n_genes))
    affected = rng.choice(n_genes, size=int(0.15 * n_genes), replace=False)
    batch_fx[:, affected] = rng.normal(0, 0.4, size=(n_batches, len(affected)))

    size_factor = np.exp(rng.normal(0, 0.25, size=n_samples))
    log2_mu = (np.log2(base)[None, :] + y[:, None] * lfc[None, :]
               + batch_fx[batch])
    mu = (2 ** log2_mu) * size_factor[:, None]

    r = 1.0 / disp
    lam = rng.gamma(shape=r[None, :], scale=mu / r[None, :])
    counts = rng.poisson(lam)

    libsize = counts.sum(axis=1)
    logcpm = np.log2(counts / libsize[:, None] * 1e6 + 1)

    counts_df = pd.DataFrame(counts, index=samples, columns=genes)
    counts_df["label"] = y
    expr_df = pd.DataFrame(np.round(logcpm, 3), index=samples, columns=genes)
    expr_df["label"] = y
    meta = pd.DataFrame({
        "sample_id": samples,
        "label": y,
        "condition": np.where(y == 1, "tumour", "normal"),
        "batch": [f"batch_{b + 1}" for b in batch],
        "library_size": libsize,
    })
    truth = pd.DataFrame({
        "gene": np.array(genes)[de_idx],
        "true_log2fc": np.round(lfc[de_idx], 3),
        "direction": np.where(lfc[de_idx] > 0, "higher in tumour",
                              "lower in tumour"),
        "mean_counts_baseline": np.round(base[de_idx], 1),
    })
    truth = truth.reindex(truth["true_log2fc"].abs()
                          .sort_values(ascending=False).index)
    return counts_df, expr_df, meta, truth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    counts, expr, meta, truth = generate(seed=args.seed)
    counts.to_csv(f"{args.out}/synth_counts.csv", index_label="sample_id")
    expr.to_csv(f"{args.out}/synth_expression_log2cpm.csv", index_label="sample_id")
    meta.to_csv(f"{args.out}/synth_metadata.csv", index=False)
    truth.to_csv(f"{args.out}/ground_truth_biomarkers.csv", index=False)
    print(f"Wrote 4 files to {args.out}/  "
          f"({len(meta)} samples, {expr.shape[1] - 1} genes, "
          f"{len(truth)} true biomarkers)")


if __name__ == "__main__":
    main()
