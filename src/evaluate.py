"""Metrics and plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

LABELS = ["normal", "tumour"]


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def save_performance_plots(y_true, y_pred, y_proba, out_dir):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                           display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/roc_curve.png", dpi=150)
    plt.close(fig)


def save_biomarker_plot(table, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    t = table.iloc[::-1]
    ax.barh(t.index, t["selection_frequency"])
    ax.set_xlabel("Selection frequency across resamples")
    ax.set_title("Top biomarker candidates")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/top_biomarkers.png", dpi=150)
    plt.close(fig)
