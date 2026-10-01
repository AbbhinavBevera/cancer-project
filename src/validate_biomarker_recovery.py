"""Check how many known true biomarkers the pipeline recovered.

Only meaningful on the SYNTHETIC dataset, where the truth is known.

    python src/validate_biomarker_recovery.py \
        --found results/top_biomarkers.csv \
        --truth data/synthetic/ground_truth_biomarkers.csv
"""
import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--found", default="results/top_biomarkers.csv")
    p.add_argument("--truth", default="data/synthetic/ground_truth_biomarkers.csv")
    args = p.parse_args()

    found = pd.read_csv(args.found, index_col=0)
    truth = pd.read_csv(args.truth).set_index("gene")

    hits = [g for g in found.index if g in truth.index]
    right_dir = [g for g in hits
                 if found.loc[g, "direction"] == truth.loc[g, "direction"]]
    print(f"Genes reported: {len(found)}")
    print(f"True biomarkers among them: {len(hits)} "
          f"(precision {len(hits) / len(found):.2f})")
    print(f"Correct direction of effect: {len(right_dir)} of {len(hits)}")
    print(f"Recall of all {len(truth)} true biomarkers: "
          f"{len(hits) / len(truth):.2f}")


if __name__ == "__main__":
    main()
