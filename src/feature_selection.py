"""Model pipelines and biomarker (gene) ranking.

Feature selection lives INSIDE the sklearn Pipeline, so during
cross-validation it is re-fitted on each training fold only. Doing it on the
full dataset first would leak information from the test data and inflate
the scores, which is a very common mistake with high-dimensional data.
"""
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# scikit-learn 1.8+ deprecates penalty in LogisticRegression; 1.9+ deprecates
# probability in SVC. Filtering keeps code clean across versions.
warnings.filterwarnings("ignore", message=".*penalty.*")
warnings.filterwarnings("ignore", message=".*probability.*")

MODEL_NAMES = ["Logistic Regression (L1)", "Linear SVM", "Random Forest"]


def build_pipeline(name, k, seed=42):
    if name == "Logistic Regression (L1)":
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                 random_state=seed)
    elif name == "Linear SVM":
        clf = SVC(kernel="linear", probability=True, random_state=seed)
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", clf),
    ])


def biomarker_stability(X, y, k, seed=42, n_splits=5, n_repeats=10, top=15):
    """Rank genes by how consistently they matter across resamples.

    For each training fold we fit an L1 logistic regression and record which
    genes get a non-zero coefficient. Genes chosen in most folds are more
    trustworthy biomarker candidates than genes that appear once.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    picked = pd.Series(0, index=X.columns, dtype=float)
    coef_sum = pd.Series(0.0, index=X.columns)
    n_fits = 0
    for train_idx, _ in cv.split(X, y):
        pipe = build_pipeline("Logistic Regression (L1)", k, seed)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        genes = X.columns[pipe.named_steps["select"].get_support()]
        coef = pd.Series(pipe.named_steps["clf"].coef_.ravel(), index=genes)
        picked[coef[coef != 0].index] += 1
        coef_sum[genes] += coef
        n_fits += 1

    table = pd.DataFrame({
        "selection_frequency": picked / n_fits,
        "mean_coefficient": coef_sum / n_fits,
    })
    table["direction"] = table["mean_coefficient"].apply(
        lambda c: "higher in tumour" if c > 0 else "lower in tumour")
    table["abs_coef"] = table["mean_coefficient"].abs()
    table = table.sort_values(["selection_frequency", "abs_coef"],
                              ascending=False).drop(columns="abs_coef")
    return table.head(top)
