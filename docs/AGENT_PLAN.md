# Implementation plan for AI agents

**Project:** Predictive Modeling and Biomarker Discovery in Cancer Transcriptomics Using Machine Learning
**Goal:** turn this repo into a rigorous, reproducible, honestly documented project on the owner's REAL dataset, ready for a human verifier to review.

Work the phases in order. Each task has an ID, outputs, and acceptance criteria. Log every completed task in `docs/AGENT_LOG.md`.

---

## 1. Non-negotiable rules

| # | Rule |
|---|------|
| R1 | **Never present synthetic results as real.** Anything derived from `data/synthetic/` is labelled "synthetic" everywhere it appears. |
| R2 | **Never invent** numbers, findings about genes, citations, dataset details, or verification/certificate claims. If unknown, leave a `[ ]` placeholder and add the item to Open Questions. |
| R3 | **Every number in the README or release notes traces to a file in `results/`**, produced by a logged command, with a row in `docs/CLAIMS_EVIDENCE.md`. |
| R4 | **No leakage.** Anything that uses labels (feature selection, tuning, supervised scaling/imputation, batch correction using labels) is fitted on training folds only. The test set is used once, at the end of Phase 4. |
| R5 | **No raw patient-level data in the repo.** Respect the dataset's licence and access terms. No personal identifiers. |
| R6 | **Do not push, tag, publish or release.** Prepare everything; the owner publishes. |
| R7 | **The owner must be able to explain every line.** Prefer simple, readable code. Comment the *why*. Summarise each change in plain English in the log. |
| R8 | **Reproducible:** fixed seeds, pinned versions, one command per stage. |
| R9 | **No clinical claims.** This is an educational/research project, not a diagnostic tool. |
| R10 | **Stop and ask the owner** if: the real dataset is not identified; the label definition is unclear; there are fewer than about 30 samples in either class; leakage is suspected; or results look too good (AUC >= 0.99 on real data). Investigate before reporting. |

---

## 2. Open questions for the owner (answer before Phase 1)

1. Which dataset was used (name, source URL, accession, licence)? Do you have the files?
2. What exactly does the label mean (tumour vs adjacent normal? subtype? stage?)
3. What did you actually do during the internship (methods, tools, results)? The repo must match this truth.
4. Who is verifying the work, and what will they check?
5. What is your deadline for the applications?

---

## 3. Current state of the repo

| Path | Status |
|------|--------|
| `src/preprocessing.py` | Done: unsupervised cleaning, optional log2 |
| `src/feature_selection.py` | Done: 3 model pipelines with selection inside each fold; stability-based biomarker ranking |
| `src/evaluate.py`, `src/train.py` | Done: metrics, plots, full run |
| `src/make_synthetic_dataset.py`, `src/validate_biomarker_recovery.py` | Done: synthetic data with known truth; recovery check |
| `data/synthetic/` | Done: 4 files + data card |
| `README.md` | Template with `[ ]` placeholders |

**Known gaps to close** (each maps to tasks below):

| Gap | Task |
|-----|------|
| G1 No real data yet | T1.x |
| G2 README placeholders unfilled | T7.x |
| G3 `notebooks/` empty, no EDA | T2.x |
| G4 No automated tests, no CI | T6.x |
| G5 Count normalisation only via `--log-transform` (no library-size correction) | T3.1 |
| G6 Hyperparameters not tuned; no confidence intervals | T4.x |
| G7 No differential-expression comparison for biomarkers | T5.1 |
| G8 No literature check of top genes | T5.4 |
| G9 No certificate or verifier evidence pack | T8.x |

---

## 4. Suggested roles

Split work across agents or sessions, but keep an independent reviewer.

| Role | Owns |
|------|------|
| Data agent | Phases 1-3 |
| Modeling agent | Phases 4-5 |
| Test/CI agent | Phase 6 |
| Docs agent | Phases 7-8 |
| Reviewer agent | Reviews each phase's output against its acceptance criteria and rules R1-R10. **Must not review its own work.** |

---

## 5. Phases and tasks

### Phase 0: Orientation (about 30 minutes)

| ID | Task | Acceptance criteria |
|----|------|--------------------|
| T0.1 | Create a virtual environment and install `requirements.txt`. | `python -c "import sklearn, pandas"` works. |
| T0.2 | Smoke test on synthetic data: `python src/train.py --data data/synthetic/synth_expression_log2cpm.csv --out results/synthetic_run` then `python src/validate_biomarker_recovery.py --found results/synthetic_run/top_biomarkers.csv`. | Results are within about 0.03 of the baseline in `data/synthetic/DATA_CARD.md` (L1 CV AUC about 0.94, SVM about 0.89, RF about 0.98; precision at 15 of at least 0.7). |
| T0.3 | Read every file in `src/` and write a plain-English summary of the pipeline into the log. | Summary states where feature selection happens and why that prevents leakage. |

### Phase 1: Real data (owner-dependent)

| ID | Task | Acceptance criteria |
|----|------|--------------------|
| T1.1 | Owner supplies dataset details (Open Question 1-2). Agent records them in `docs/DATA_CARD_real.md` (source, accession, licence, collection notes, label definition, what was excluded). | No field filled with a guess. Unknowns marked `[unknown]`. |
| T1.2 | Write `src/prepare_real_data.py`: converts the downloaded files to `data/expression.csv` (rows = samples, columns = genes, last column `label`) and `data/metadata.csv` (batch/site/age/sex if available). | One command; deterministic; documented in `data/README.md`. |
| T1.3 | Write `src/validate_data.py` with checks: unique sample IDs, unique gene IDs, numeric values, no missing labels, both classes present, no duplicate samples, label not recoverable from sample ID text. | Script exits non-zero on failure; prints a summary table. |
| T1.4 | Confirm `.gitignore` keeps raw/real data out of Git. | `git status` shows no data files staged. |

### Phase 2: Quality control and exploration

Create `notebooks/01_data_exploration.ipynb`.

| ID | Task | Acceptance criteria |
|----|------|--------------------|
| T2.1 | Class balance, library-size distribution by class, zero-inflation, missingness. | Plots plus a 3-5 sentence written interpretation. |
| T2.2 | PCA (and optionally UMAP) coloured by label AND by batch/site. | If batch drives PC1 more than label, flag it in the log. |
| T2.3 | Outlier samples: identify and report. **Do not drop silently.** | Any exclusion is listed with reason in the log and `DATA_CARD_real.md`. |
| T2.4 | Confounder check: cross-tab and chi-square/ANOVA of batch, site, age, sex vs label. | Result recorded. If a confounder is strongly associated with the label, add it to Limitations. |

### Phase 3: Preprocessing

| ID | Task | Acceptance criteria |
|----|------|--------------------|
| T3.1 | If data is raw counts, add a `--normalize` option to `preprocessing.py` (`none`, `logcpm`). Keep it unsupervised. | Synthetic counts file gives results similar to the log2CPM file. |
| T3.2 | Justify the gene-filter threshold (for example, minimum counts in at least N samples). | Rationale written in the notebook; threshold is a CLI argument. |
| T3.3 | Batch handling: compare (a) ignoring batch, (b) including batch as a covariate, (c) correction fitted on training folds only. | Comparison table of CV metrics; choice justified. No label-using correction on the full dataset. |
| T3.4 | Add a check or test that preprocessing never reads labels. | Test passes (see T6.1). |

### Phase 4: Modeling and validation

| ID | Task | Acceptance criteria |
|----|------|--------------------|
| T4.1 | **Freeze the test split:** save sample IDs of the held-out test set to `results/splits.json` and never touch it until T4.6. | File exists; seed recorded. |
| T4.2 | Tune `k` (genes kept), `C`, and random-forest depth using CV on the training set only (nested CV preferred). | Search space and best values logged. |
| T4.3 | Report metrics with 95% bootstrap confidence intervals. | `results/cv_results.csv` includes intervals. |
| T4.4 | **Sanity checks:** (a) label-permutation test, where shuffled labels must give AUC near 0.5; (b) a deliberately leaky version (select genes on all data first) to show how much scores inflate; (c) a trivial baseline (majority class). | All three reported. If (a) is far above 0.5, stop: there is leakage or a confounder. |
| T4.5 | Calibration curve and threshold discussion. Recall is prioritised because missing a real case is costlier. | Plot saved; choice explained in words. |
| T4.6 | **One-time** evaluation on the frozen test set with the final chosen model. | Test set touched exactly once; noted in the log. |
| T4.7 | If real-data AUC >= 0.99, investigate before reporting (leakage, duplicates, batch = label). | Written explanation in the log. |

### Phase 5: Biomarker analysis

| ID | Task | Acceptance criteria |
|----|------|--------------------|
| T5.1 | Add a univariate differential-expression analysis (Mann-Whitney or t-test per gene, effect size, Benjamini-Hochberg FDR correction) alongside the model-based ranking. | `results/differential_expression.csv` |
| T5.2 | Compare the two rankings: overlap, and genes found by only one method. | Short written interpretation. |
| T5.3 | On **synthetic** data only: measure recall and precision against `ground_truth_biomarkers.csv`. Try to improve recall (currently about 30% in top 15) by reporting a longer list; explain the trade-off. | Numbers in the log, labelled synthetic. |
| T5.4 | Literature check for the top real genes. Record each source in `docs/literature_notes.md`: title, authors, year, DOI or PMID, and the exact claim it supports. | **Only cite sources you actually opened.** If a source cannot be verified, write "not verified". Never cite from memory. |
| T5.5 | Optional: pathway enrichment with multiple-testing correction. | Clearly labelled exploratory. |

### Phase 6: Tests, CI and reproducibility

| ID | Task | Acceptance criteria |
|----|------|--------------------|
| T6.1 | Add `tests/test_pipeline.py` (pytest): (a) the selector is a step inside the sklearn Pipeline; (b) same seed gives identical results; (c) preprocessing does not use labels; (d) on synthetic data, top-15 biomarker precision is at least 0.6. | `pytest -q` passes. |
| T6.2 | Add `.github/workflows/ci.yml` that installs requirements and runs pytest on the synthetic data. | Workflow file valid; owner enables it by pushing. |
| T6.3 | Pin package versions in `requirements.txt`. | Fresh environment reproduces the baseline. |
| T6.4 | Add `run_all.sh` (or Makefile) running each stage in order. | One command reproduces `results/`. |

### Phase 7: Documentation

| ID | Task | Acceptance criteria |
|----|------|--------------------|
| T7.1 | Fill README placeholders **only** from `results/` files; add a row per number to `docs/CLAIMS_EVIDENCE.md`. | Zero unfilled `[ ]` left, or the section is deleted. |
| T7.2 | Save final figures to `results/` (ROC, confusion matrix, biomarkers, PCA). | Files exist and are referenced in the README. |
| T7.3 | Write the Limitations section from what was actually observed (sample size, batch, single cohort, no external validation). | Every limitation traces to a finding in the log. |
| T7.4 | Write `docs/EXPLAINER.md`: plain-English walkthrough of every step, plus 15 questions a verifier or interviewer might ask, with honest answers. | Owner reviews and edits it in their own words. |
| T7.5 | Add `CITATION.cff` with the owner's details. | Valid file. |

### Phase 8: Release preparation (owner publishes)

| ID | Task | Acceptance criteria |
|----|------|--------------------|
| T8.1 | Run the definition-of-done checklist (Section 7). | All items ticked or explicitly waived by the owner. |
| T8.2 | Fill `RELEASE_NOTES` from real results. | Every number appears in `CLAIMS_EVIDENCE.md`. |
| T8.3 | Build the verifier evidence pack: README, `CLAIMS_EVIDENCE.md`, `results/`, `DATA_CARD_real.md`, `literature_notes.md`. | Zip created. **Owner** pushes, tags and releases. |

### Phase 9: Optional extensions (only after Phase 8)

External validation on a second public cohort; SHAP or permutation importance; subtype classification; comparison with published signatures (with citations); a small Streamlit demo (clearly marked "not for clinical use").

---

## 6. Logging protocol

After every task, append an entry to `docs/AGENT_LOG.md` (format is in that file). Every entry must state whether real or synthetic data was used. Reviewer agents sign off phase by phase in the same file.

---

## 7. Definition of done

- [ ] Pipeline runs end to end on the real dataset with one documented command
- [ ] No leakage: selection and tuning inside training folds; test set used once
- [ ] Permutation-test sanity check passed (AUC near 0.5 on shuffled labels)
- [ ] Metrics reported with confidence intervals
- [ ] Biomarkers compared across two methods; literature claims all traceable to opened sources
- [ ] README has no placeholders; every number is in `CLAIMS_EVIDENCE.md`
- [ ] Limitations section is honest and specific
- [ ] `pytest` passes; CI workflow present
- [ ] Synthetic material clearly labelled everywhere it appears
- [ ] Owner has read `EXPLAINER.md` and can explain each step unaided
- [ ] No raw data, secrets, or personal identifiers in the repo
- [ ] Reviewer agent and owner sign-off recorded in the log

---

## 8. Verifier evidence pack: what a reviewer should be able to check

| Claim | Where to look |
|-------|---------------|
| Dataset and label definition | `docs/DATA_CARD_real.md` |
| Preprocessing choices | `notebooks/01_data_exploration.ipynb`, `src/preprocessing.py` |
| No data leakage | `src/feature_selection.py` (selector inside Pipeline), `results/` permutation-test output, `tests/` |
| Model performance | `results/cv_results.csv`, `results/test_metrics.csv` |
| Biomarker candidates | `results/top_biomarkers.csv`, `results/differential_expression.csv` |
| Literature support | `docs/literature_notes.md` |
| Everything reproducible | `run_all.sh`, pinned `requirements.txt`, CI |
| Who did what | `docs/AGENT_LOG.md` |
