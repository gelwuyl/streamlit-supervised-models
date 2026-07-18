"""Model definitions and hyperparameter grids for the assignment."""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# --- Regression models (Part 1) ---
REG_MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(max_iter=10000),
}

# Alpha grids (trimmed for speed; log-spaced so we cover small -> large).
# Both are length 7 so the Streamlit alpha slider (0..6) works for either model.
RIDGE_ALPHAS = np.logspace(-2, 2, 7)   # 0.01 ... 100
LASSO_ALPHAS = np.logspace(-2, 1, 7)   # 0.01 ... 10
RIDGE_GRID = {"alpha": RIDGE_ALPHAS}
LASSO_GRID = {"alpha": LASSO_ALPHAS}

# --- Classification models (Part 2) ---
CLF_MODELS = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

CLF_GRIDS = {
    "DecisionTree": {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
    },
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
    },
    "GradientBoosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5],
    },
}
