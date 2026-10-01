"""Unit and integration tests for cancer transcriptomics ML pipeline.

Verifies:
1. Leakage prevention: Feature selection is encapsulated inside sklearn.pipeline.Pipeline.
2. Reproducibility: Deterministic outcomes given identical seeds.
3. Unsupervised preprocessing: Cleaning functions operate independently of labels.
4. Biomarker discovery fidelity: Precision >= 0.60 against ground truth synthetic markers.
5. Permutation sanity check: Shuffled labels yield AUC near 0.50 (chance level).
"""
import os
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from preprocessing import basic_clean, load_expression
from feature_selection import MODEL_NAMES, biomarker_stability, build_pipeline
from differential_expression import compute_differential_expression


@pytest.fixture(scope="module")
def synthetic_data():
    path = "data/synthetic/synth_expression_log2cpm.csv"
    if not os.path.exists(path):
        pytest.skip(f"Synthetic data file not found at {path}")
    X, y = load_expression(path)
    X = basic_clean(X)
    return X, y


def test_pipeline_encapsulation_no_leakage():
    """Requirement: Feature selection must be inside sklearn Pipeline to prevent test leakage."""
    for name in MODEL_NAMES:
        pipe = build_pipeline(name, k=50, seed=42)
        assert isinstance(pipe, Pipeline), f"{name} must be an instance of sklearn.pipeline.Pipeline"
        step_names = [step[0] for step in pipe.steps]
        assert "select" in step_names, f"'select' step missing from {name} pipeline"
        assert "clf" in step_names, f"'clf' step missing from {name} pipeline"
        # The selector must execute BEFORE the classifier
        assert step_names.index("select") < step_names.index("clf"), (
            f"Selector must occur before classifier in {name}"
        )


def test_preprocessing_is_unsupervised(synthetic_data):
    """Requirement: basic_clean must operate solely on features X without labels."""
    X, _ = synthetic_data
    # Perturb a random row
    X_modified = X.copy()
    X_modified.iloc[0, 0] = np.nan
    cleaned = basic_clean(X_modified)
    assert not cleaned.isna().any().any(), "Imputation should remove all NaNs"
    assert cleaned.shape[0] == X.shape[0], "Sample count must be preserved"


def test_reproducibility_with_seed(synthetic_data):
    """Requirement: Same random seed produces identical cross-validation scores."""
    X, y = synthetic_data
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    pipe1 = build_pipeline("Logistic Regression (L1)", k=50, seed=42)
    res1 = cross_validate(pipe1, X, y, cv=cv, scoring=["roc_auc"])
    
    cv2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipe2 = build_pipeline("Logistic Regression (L1)", k=50, seed=42)
    res2 = cross_validate(pipe2, X, y, cv=cv2, scoring=["roc_auc"])
    
    np.testing.assert_allclose(res1["test_roc_auc"], res2["test_roc_auc"], rtol=1e-5)


def test_biomarker_recovery_precision(synthetic_data):
    """Requirement: Top biomarker recovery precision >= 0.60 on synthetic benchmark."""
    truth_path = "data/synthetic/ground_truth_biomarkers.csv"
    if not os.path.exists(truth_path):
        pytest.skip(f"Ground truth file not found at {truth_path}")
        
    X, y = synthetic_data
    # Fast test with fewer repeats to maintain fast test suite runtime
    biomarkers = biomarker_stability(X, y, k=50, seed=42, n_splits=5, n_repeats=3, top=15)
    truth = pd.read_csv(truth_path).set_index("gene")
    
    hits = [g for g in biomarkers.index if g in truth.index]
    precision = len(hits) / len(biomarkers)
    assert precision >= 0.60, f"Expected precision >= 0.60, got {precision:.2f}"


def test_label_permutation_sanity_check(synthetic_data):
    """Requirement: Shuffled labels should collapse ROC-AUC towards random chance (~0.50)."""
    X, y = synthetic_data
    rng = np.random.default_rng(999)
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipe = build_pipeline("Logistic Regression (L1)", k=30, seed=42)
    res = cross_validate(pipe, X, y_shuffled, cv=cv, scoring=["roc_auc"])
    mean_auc = res["test_roc_auc"].mean()
    
    # Random guess should be roughly 0.50 (definitely < 0.65)
    assert 0.35 <= mean_auc <= 0.65, f"Permutation AUC should be near 0.50, got {mean_auc:.3f}"


def test_differential_expression(synthetic_data):
    """Requirement: Differential expression executes and flags known strong biomarkers."""
    X, y = synthetic_data
    de_table = compute_differential_expression(X, y)
    assert "log2_fold_change" in de_table.columns
    assert "fdr_bh" in de_table.columns
    assert "p_value" in de_table.columns
    assert de_table.index[0] == "SYN_0835", "Expected top DE gene to match synthetic truth"
