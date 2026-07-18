"""GridSearchCV wrappers + shared precompute helpers.

IMPORTANT: both the notebook AND the Streamlit app call the SAME
precompute_* functions below, so they always show identical results
(no duplicated tuning, no divergence between the two deliverables).
"""
from sklearn.model_selection import GridSearchCV

from .models import (
    RIDGE_GRID, LASSO_GRID, CLF_GRIDS,
    REG_MODELS, CLF_MODELS, RIDGE_ALPHAS, LASSO_ALPHAS,
)


def run_regression_gridsearch(model, param_grid, X_train, y_train,
                              cv=5, scoring="neg_mean_squared_error"):
    gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs


def run_classification_gridsearch(model, param_grid, X_train, y_train,
                                  cv=5, scoring="f1"):
    gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs


def precompute_reg_results(X_train, y_train):
    """Run Ridge & Lasso tuning once. Returns a dict with fitted GridSearchCVs."""
    ridge_gs = run_regression_gridsearch(REG_MODELS["Ridge"], RIDGE_GRID, X_train, y_train)
    lasso_gs = run_regression_gridsearch(REG_MODELS["Lasso"], LASSO_GRID, X_train, y_train)
    return {
        "ridge": ridge_gs,
        "lasso": lasso_gs,
        "ridge_alphas": RIDGE_ALPHAS,
        "lasso_alphas": LASSO_ALPHAS,
    }


def precompute_clf_results(X_train, y_train):
    """Run the 3 classifiers' tuning once. Returns {name: fitted GridSearchCV}."""
    results = {}
    for name, model in CLF_MODELS.items():
        results[name] = run_classification_gridsearch(model, CLF_GRIDS[name], X_train, y_train)
    return results
