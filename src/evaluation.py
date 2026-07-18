"""Metrics and plotting helpers (shared by notebook and Streamlit app)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, f1_score, roc_auc_score, roc_curve,
)


def regression_metrics(y_true, y_pred):
    """Return a dict with the Mean Squared Error."""
    return {"mse": mean_squared_error(y_true, y_pred)}


def classification_metrics(model, X_test, y_test):
    """Return F1 and AUC. AUC needs probabilities, so we guard for that."""
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    auc = roc_auc_score(y_test, y_score) if y_score is not None else None
    return {"f1": f1, "auc": auc}


def plot_coefficient_paths(paths, current_alpha=None, title="Coefficient Paths"):
    """paths: {model_name: (alphas_array, list_of_coef_arrays)}."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, (alphas, coefs) in paths.items():
        coefs = np.array(coefs)            # shape (n_alphas, n_features)
        for j in range(coefs.shape[1]):
            ax.plot(alphas, coefs[:, j], alpha=0.6,
                    label=name if j == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient value")
    ax.set_title(title)
    if current_alpha is not None:
        ax.axvline(current_alpha, color="red", linestyle="--",
                   label=f"current alpha={current_alpha:.3g}")
    ax.legend()
    return fig


def plot_roc_curves(roc_data):
    """roc_data: {model_name: (y_test, y_proba)} -> overlaid ROC curves."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_test, y_proba) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- Ensemble Comparison")
    ax.legend()
    return fig
