# Start here

This takeaway pack contains:

- **A working ML project** for tumour-vs-normal prediction and biomarker discovery from gene expression (`src/`)
- **A synthetic dataset** with known right answers (`data/synthetic/`), clearly NOT real patient data
- **An implementation plan for AI agents** (`docs/AGENT_PLAN.md`, with `AGENTS.md` as the short version)
- Templates for logging work and tracing every claim to evidence (`docs/AGENT_LOG.md`, `docs/CLAIMS_EVIDENCE.md`)

## Try it in 2 minutes

```bash
pip install -r requirements.txt
python src/train.py --data data/synthetic/synth_expression_log2cpm.csv
python src/validate_biomarker_recovery.py
```

## Before putting anything on GitHub or in an application

1. Replace the synthetic data with your real dataset (Phase 1 of the plan).
2. Run the pipeline on the real data and fill in the README from the real outputs only.
3. Delete or clearly label anything that came from the synthetic data.
4. Make sure the code matches what you actually did, and that you can explain it.
5. You publish. Agents prepare the work but never push or release.
