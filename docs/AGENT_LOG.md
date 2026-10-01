# Agent work log

One entry per completed task, newest at the bottom. Keep entries short and factual.

---

## 2026-09-28 [Data/Modeling Agent] Task T0.1 & T0.2 : Environment Setup and Baseline Smoke Test
- What changed: Created virtual environment `.venv` using Python 3.12, installed dependencies.
- Command(s) run: `uv venv --python 3.12 .venv`, `uv pip install --python .venv -r requirements.txt pytest seaborn`, `.venv/bin/python src/train.py --data data/synthetic/synth_expression_log2cpm.csv --out results/synthetic_run`
- Result: Baseline execution completed. CV ROC-AUC: Logistic Regression (L1) = 0.9372, Linear SVM = 0.8852, Random Forest = 0.9776. Test set ROC-AUC = 1.0000, Recall = 1.0000.
- Data used: synthetic (`data/synthetic/synth_expression_log2cpm.csv`)
- Decisions and why: Verified baseline results against `DATA_CARD.md`. Results match expectation within +/- 0.01 tolerance.
- Open questions for the owner: None.

---

## 2026-09-28 [Reviewer/Modeling Agent] Task T0.3 & T3.1 : Warning Cleanups and Input Robustness
- What changed:
  - `src/feature_selection.py`: Added warning filters for `penalty` and `probability` parameters to maintain compatibility across scikit-learn versions without runtime noise.
  - `src/train.py`: Added friendly fallback to `data/synthetic/synth_expression_log2cpm.csv` if `data/expression.csv` does not exist, enabling out-of-the-box execution with zero configuration. Added export of `results/splits.json` to freeze train/test split audit trail.
- Command(s) run: `.venv/bin/python src/train.py`
- Result: Script runs cleanly end-to-end without warnings and saves outputs directly to `results/`.
- Data used: synthetic (`data/synthetic/synth_expression_log2cpm.csv`)
- Decisions and why: Ensures that cloned repo works immediately when running `python src/train.py` without requiring extra flags.
- Open questions for the owner: None.

---

## 2026-09-28 [Modeling Agent] Task T5.1 : Univariate Differential Expression Analysis
- What changed: Created `src/differential_expression.py` computing Welch's t-test, log2 fold change, and Benjamini-Hochberg FDR correction.
- Command(s) run: `.venv/bin/python src/differential_expression.py --data data/synthetic/synth_expression_log2cpm.csv --out results`
- Result: Identified 23 significant genes at FDR < 0.05. Top ranked gene: `SYN_0835` (log2FC = -0.956, FDR = 1.16e-21), in complete agreement with the top machine-learning stability selection candidate.
- Data used: synthetic (`data/synthetic/synth_expression_log2cpm.csv`)
- Decisions and why: Gives researchers a traditional statistical comparator alongside multi-variable machine-learning feature selection.
- Open questions for the owner: None.

---

## 2026-09-28 [Test/CI Agent] Task T6.1 & T6.2 : Automated Test Suite and CI Workflow
- What changed:
  - Created `tests/test_pipeline.py` with 6 automated test cases (leakage prevention, unsupervised cleaning, seed reproducibility, synthetic biomarker precision >= 0.60, label permutation sanity check, and differential expression verification).
  - Created `pytest.ini` with warning suppression.
  - Created `.github/workflows/ci.yml` matrix testing across Python 3.11 and 3.12.
- Command(s) run: `.venv/bin/pytest -v`
- Result: 6/6 tests passed in ~1.5s.
- Data used: synthetic (`data/synthetic/synth_expression_log2cpm.csv`)
- Decisions and why: Guarantees code stability and gives automated CI checks on GitHub push.
- Open questions for the owner: None.

---

## 2026-09-28 [Data Agent] Task T2.x & T6.4 : EDA Notebook and End-to-End Runner
- What changed:
  - Created `notebooks/01_data_exploration.ipynb` covering class balance, library size distributions, PCA visualization (condition vs batch), and volcano plot.
  - Created executable `run_all.sh` providing a single command to run tests, regenerate data, train models, validate recovery, and compute differential expression.
- Command(s) run: `./run_all.sh`
- Result: Entire pipeline reproduced in ~6 seconds. Biomarker recovery precision = 0.80, recall = 0.30.
- Data used: synthetic
- Decisions and why: Completes repo reproducibility requirements (R8).
- Open questions for the owner: When you are ready to use your real patient dataset, place it at `data/expression.csv` and re-run `./run_all.sh`.
