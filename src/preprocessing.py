"""Loading and cleaning the expression matrix."""
import numpy as np
import pandas as pd


def load_expression(path, label_col="label"):
    """Return (X, y): X = samples x genes, y = 0/1 labels."""
    df = pd.read_csv(path, index_col=0)
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in {path}")
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])
    return X, y


def basic_clean(X, min_mean_expr=0.1, log_transform=False):
    """Unsupervised cleaning only (no labels used, so no leakage).

    - drop genes missing in more than 10% of samples
    - fill remaining gaps with the gene median
    - drop genes with very low average expression
    - optionally log2(x + 1) transform (use for raw counts)
    """
    X = X.dropna(axis=1, thresh=int(0.9 * len(X)))
    X = X.fillna(X.median())
    X = X.loc[:, X.mean() >= min_mean_expr]
    if log_transform:
        X = np.log2(X + 1)
    return X
