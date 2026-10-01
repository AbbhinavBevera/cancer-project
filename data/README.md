# Data

Raw data is not stored in this repository.

## Expected format

A CSV file at `data/expression.csv`:

- one row per sample, first column = sample ID
- one column per gene (expression values)
- a final column named `label` (0 = normal, 1 = tumour)

## Using your real dataset

Download the expression data from the source you used
([dataset name and link, e.g. TCGA-BRCA via GDC / GEO accession]),
reshape it to the format above, and save it as `data/expression.csv`.

## Trying the pipeline without real data

`data/synthetic/` contains a **synthetic** tumour-vs-normal dataset with known
true biomarkers. Read `data/synthetic/DATA_CARD.md` first. Regenerate it with
`python src/make_synthetic_dataset.py --out data/synthetic`.
Results from synthetic data mean nothing biologically. Never report them as real findings.
