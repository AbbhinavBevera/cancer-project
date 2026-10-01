"""Export trained model parameters, test presets, and metrics for client-side web UI."""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from preprocessing import basic_clean, load_expression
from feature_selection import build_pipeline


def main():
    data_path = "data/synthetic/synth_expression_log2cpm.csv"
    X, y = load_expression(data_path)
    X = basic_clean(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    k = min(200, X.shape[1])
    pipe_lr = build_pipeline("Logistic Regression (L1)", k=k, seed=42)
    pipe_lr.fit(X_train, y_train)

    scaler = pipe_lr.named_steps["scale"]
    selector = pipe_lr.named_steps["select"]
    clf = pipe_lr.named_steps["clf"]

    support = selector.get_support()
    selected_indices = np.where(support)[0]
    selected_genes = [X.columns[i] for i in selected_indices]

    raw_coefs = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    features = {}
    for i, gene in enumerate(selected_genes):
        c = float(raw_coefs[i])
        orig_idx = selected_indices[i]
        m = float(scaler.mean_[orig_idx])
        s = float(scaler.scale_[orig_idx])
        # Only retain non-zero coefficients
        if abs(c) > 1e-5:
            features[gene] = {
                "coef": round(c, 5),
                "mean": round(m, 4),
                "scale": round(s, 4),
                "min": round(float(X[gene].min()), 2),
                "max": round(float(X[gene].max()), 2),
                "normal_mean": round(float(X.loc[y == 0, gene].mean()), 3),
                "tumour_mean": round(float(X.loc[y == 1, gene].mean()), 3),
            }

    # Load top biomarkers table
    biomarkers_df = pd.read_csv("results/top_biomarkers.csv", index_col=0)
    top_biomarkers = []
    for gene, row in biomarkers_df.head(15).iterrows():
        top_biomarkers.append({
            "gene": gene,
            "frequency": float(row["selection_frequency"]),
            "mean_coefficient": float(row["mean_coefficient"]),
            "direction": str(row["direction"]),
            "min": float(features[gene]["min"]) if gene in features else 0.0,
            "max": float(features[gene]["max"]) if gene in features else 15.0,
            "normal_mean": float(features[gene]["normal_mean"]) if gene in features else 0.0,
            "tumour_mean": float(features[gene]["tumour_mean"]) if gene in features else 0.0,
        })

    # Test presets (mix of obvious tumour, obvious normal, and borderline cases)
    lr_probs = pipe_lr.predict_proba(X_test)[:, 1]
    presets = []

    # Pick samples across probability range
    sample_indices = list(X_test.index)
    sorted_by_prob = np.argsort(lr_probs)

    # 3 clear normal, 2 borderline, 3 clear tumour
    picks = [
        ("Normal Tissue (Low Risk)", sorted_by_prob[1]),
        ("Normal Tissue (Standard)", sorted_by_prob[5]),
        ("Normal Tissue (Variable)", sorted_by_prob[10]),
        ("Ambiguous Case (Borderline)", sorted_by_prob[len(sorted_by_prob)//2 - 1]),
        ("Intermediate Risk Case", sorted_by_prob[len(sorted_by_prob)//2 + 2]),
        ("Tumour (Early Signature)", sorted_by_prob[-12]),
        ("Malignant Tumour (High Risk)", sorted_by_prob[-4]),
        ("Malignant Tumour (Aggressive)", sorted_by_prob[-1]),
    ]

    for label_desc, p_idx in picks:
        sample_id = sample_indices[p_idx]
        row = X_test.loc[sample_id]
        true_y = int(y_test.loc[sample_id])
        prob = float(lr_probs[p_idx])
        
        gene_values = {g: round(float(row[g]), 3) for g in features.keys()}
        presets.append({
            "description": f"{sample_id}: {label_desc}",
            "sample_id": sample_id,
            "true_label": true_y,
            "true_condition": "Tumour" if true_y == 1 else "Normal",
            "model_probability": round(prob, 4),
            "genes": gene_values,
        })

    # Load metrics
    cv_res = pd.read_csv("results/cv_results.csv", index_col=0).to_dict(orient="index")
    test_met = pd.read_csv("results/test_metrics.csv", index_col=0)["value"].to_dict()

    model_data = {
        "intercept": round(intercept, 5),
        "features": features,
        "top_biomarkers": top_biomarkers,
        "presets": presets,
        "metrics": {
            "cv": cv_res,
            "test": test_met,
        },
        "metadata": {
            "title": "Cancer Transcriptomics Biomarker & Prediction Engine",
            "features_count": len(features),
            "generated_at": "2026-09-28",
        }
    }

    # Write output to web/ and docs/
    for out_dir in ["web", "docs"]:
        os.makedirs(out_dir, exist_ok=True)
        js_path = os.path.join(out_dir, "model_data.js")
        with open(js_path, "w") as f:
            f.write("// Auto-generated client-side model parameters\n")
            f.write("window.MODEL_DATA = ")
            f.write(json.dumps(model_data, indent=2))
            f.write(";\n")
        print(f"Wrote model data to {js_path}")


if __name__ == "__main__":
    main()
