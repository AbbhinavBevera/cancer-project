"""Run the full pipeline: load -> clean -> compare models -> test -> biomarkers.

Example:
    python src/train.py --data data/expression.csv --k 200
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)

from evaluate import compute_metrics, save_biomarker_plot, save_performance_plots
from feature_selection import MODEL_NAMES, biomarker_stability, build_pipeline
from preprocessing import basic_clean, load_expression


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/expression.csv")
    p.add_argument("--label-col", default="label")
    p.add_argument("--k", type=int, default=200, help="genes kept by selector")
    p.add_argument("--log-transform", action="store_true",
                   help="apply log2(x+1); use for raw counts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. load + clean
    if args.data == "data/expression.csv" and not os.path.exists(args.data):
        synth_path = "data/synthetic/synth_expression_log2cpm.csv"
        if os.path.exists(synth_path):
            print(f"[Notice] '{args.data}' not found. Defaulting to demonstration data: '{synth_path}'.")
            args.data = synth_path

    X, y = load_expression(args.data, args.label_col)
    X = basic_clean(X, log_transform=args.log_transform)
    k = min(args.k, X.shape[1])
    print(f"Samples: {X.shape[0]}  Genes after cleaning: {X.shape[1]}  "
          f"Class counts: {y.value_counts().to_dict()}")

    # 2. hold out a test set that is never touched until the end
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    # Save frozen splits for auditability
    import json
    with open(f"{args.out}/splits.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "train_samples": list(X_train.index),
            "test_samples": list(X_test.index),
        }, f, indent=2)

    # 3. compare models with cross-validation (selection happens per fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    rows = []
    for name in MODEL_NAMES:
        res = cross_validate(build_pipeline(name, k, args.seed), X_train, y_train,
                             cv=cv, scoring=["accuracy", "recall", "roc_auc"])
        rows.append({
            "model": name,
            "cv_accuracy": res["test_accuracy"].mean(),
            "cv_accuracy_std": res["test_accuracy"].std(),
            "cv_recall": res["test_recall"].mean(),
            "cv_recall_std": res["test_recall"].std(),
            "cv_roc_auc": res["test_roc_auc"].mean(),
            "cv_roc_auc_std": res["test_roc_auc"].std(),
        })
    cv_table = pd.DataFrame(rows).set_index("model").round(4)
    print("\nCross-validation results (training set only):")
    print(cv_table[["cv_accuracy", "cv_recall", "cv_roc_auc"]])
    cv_table.to_csv(f"{args.out}/cv_results.csv")

    # 4. pick the best model by CV ROC-AUC, evaluate once on the test set
    best_name = cv_table["cv_roc_auc"].idxmax()
    best = build_pipeline(best_name, k, args.seed).fit(X_train, y_train)
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print(f"\nBest model: {best_name}")
    print("Test set results:")
    for name, value in metrics.items():
        print(f"  {name:10s} {value:.4f}")
    pd.Series(metrics).round(4).to_csv(f"{args.out}/test_metrics.csv",
                                       header=["value"])
    save_performance_plots(y_test, y_pred, y_proba, args.out)
    joblib.dump(best, f"{args.out}/best_model.joblib")

    # 5. biomarker candidates (stability across resamples, training data only)
    biomarkers = biomarker_stability(X_train, y_train, k, args.seed)
    biomarkers.round(4).to_csv(f"{args.out}/top_biomarkers.csv")
    save_biomarker_plot(biomarkers, args.out)
    print("\nTop biomarker candidates:")
    print(biomarkers.round(3).head(10))
    print(f"\nAll outputs saved to {args.out}/")


if __name__ == "__main__":
    main()
