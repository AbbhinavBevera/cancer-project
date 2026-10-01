# Claims and evidence

Every claim in the README, release notes, or applications must have a row here.
If there is no evidence file, the claim does not go in.

| # | Claim (as written in README) | Evidence file | Command that produced it | Real or synthetic data | Verified by |
|---|------------------------------|---------------|--------------------------|------------------------|-------------|
| 1 | Logistic Regression CV ROC-AUC = 0.9372 | results/cv_results.csv | `python src/train.py` | synthetic | Automated Run (2026-09-28) |
| 2 | Linear SVM CV ROC-AUC = 0.8852 | results/cv_results.csv | `python src/train.py` | synthetic | Automated Run (2026-09-28) |
| 3 | Random Forest CV ROC-AUC = 0.9776 | results/cv_results.csv | `python src/train.py` | synthetic | Automated Run (2026-09-28) |
| 4 | Best model test set recall = 1.0000 | results/test_metrics.csv | `python src/train.py` | synthetic | Automated Run (2026-09-28) |
| 5 | Best model test set accuracy = 0.9167 | results/test_metrics.csv | `python src/train.py` | synthetic | Automated Run (2026-09-28) |
| 6 | Top candidate SYN_0835 frequency = 1.00 | results/top_biomarkers.csv | `python src/train.py` | synthetic | Automated Run (2026-09-28) |
| 7 | Biomarker precision = 0.80 (12/15) | terminal output | `python src/validate_biomarker_recovery.py` | synthetic | Automated Run (2026-09-28) |
| 8 | Differential expression identifies 23 significant genes (FDR < 0.05) | results/differential_expression.csv | `python src/differential_expression.py` | synthetic | Automated Run (2026-09-28) |
| 9 | Permutation test collapses AUC to chance (~0.50) | tests/test_pipeline.py | `pytest -q` | synthetic | Automated Run (2026-09-28) |
