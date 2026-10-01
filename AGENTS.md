# Instructions for AI agents

Read `docs/AGENT_PLAN.md` before doing anything. It has the full plan, task IDs,
and acceptance criteria. The short version:

1. **Never present synthetic results as real.** Anything from `data/synthetic/` must be labelled synthetic.
2. **Never invent** numbers, citations, gene findings, dataset details, or verification claims. Use `[ ]` placeholders and ask the owner.
3. **No data leakage.** Anything that uses labels is fitted on training folds only. The test set is used once.
4. **Every number in the README must trace to a file in `results/`** and a row in `docs/CLAIMS_EVIDENCE.md`.
5. **Do not push, tag, or publish.** Prepare work; the owner publishes.
6. **Keep code simple and commented.** The owner must be able to explain every line.
7. **Log your work** in `docs/AGENT_LOG.md` after each task.
8. **Stop and ask** if the dataset is unknown, results look too good (AUC >= 0.99 on real data), or leakage is suspected.

Quick checks:

```bash
pip install -r requirements.txt
python src/make_synthetic_dataset.py --out data/synthetic
python src/train.py --data data/synthetic/synth_expression_log2cpm.csv --out results/synthetic_run
python src/validate_biomarker_recovery.py --found results/synthetic_run/top_biomarkers.csv
```
