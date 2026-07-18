"""ML Playground — interactive Streamlit app for the NTU assignment.

Run locally:  streamlit run streamlit_app.py
Reuses the SAME src/ logic as the notebook, so results match exactly.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src import data_loader, models, training, evaluation

st.set_page_config(page_title="ML Playground", layout="wide")
st.sidebar.title("ML Playground")
page = st.sidebar.radio("Choose a part", ["Regularization", "Ensemble Methods"])

# ---------------- Regularization page ----------------
if page == "Regularization":
    st.title("Part 1 — Regularization (diabetes)")
    st.markdown(
        "**Regularization**"
        "\n""- Use the `diabetes` dataset from `sklearn.datasets`."
        "\n""- Compare the performance (Mean Squared Error) of `LinearRegression`, `Ridge`, and `Lasso` models."
        "\n""- Tune the `alpha` parameter for `Ridge` and `Lasso` using `GridSearchCV` with cross-validation to find the optimal regularization strength."
        "\n"
        "```python\n"
        "from sklearn.datasets import load_diabetes\n"
        "\n"
        "# Load the diabetes dataset\n"
        "diabetes = load_diabetes()\n"
        "```"
    )
    st.markdown("Pick a model and an `alpha`, and watch the test MSE and coefficient paths update live.")

    @st.cache_resource
    def get_reg_data():
        X_train, X_test, y_train, y_test, scaler, feat = data_loader.load_diabetes_split()
        reg = training.precompute_reg_results(X_train, y_train)
        return X_train, X_test, y_train, y_test, reg

    X_train, X_test, y_train, y_test, reg = get_reg_data()

    model_name = st.sidebar.selectbox("Model", ["LinearRegression", "Ridge", "Lasso"])
    alpha_index = st.sidebar.slider(
        "Alpha (index into tuned grid)", 0, len(reg["ridge_alphas"]) - 1, 3,
        help="Alpha is log-spaced; the label below shows the actual value.")
    alphas = reg["ridge_alphas"] if model_name != "Lasso" else reg["lasso_alphas"]
    alpha = float(alphas[alpha_index])
    st.sidebar.write(f"Selected alpha = **{alpha:.4g}**")

    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=alpha)
    else:
        model = Lasso(alpha=alpha, max_iter=10000)

    with st.spinner("Fitting..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

    st.metric("Test MSE", f"{mse:.2f}")
    st.write(
        f"Best tuned alpha — Ridge: {reg['ridge'].best_params_['alpha']:.4g} | "
        f"Lasso: {reg['lasso'].best_params_['alpha']:.4g}")

    ridge_coefs, lasso_coefs = [], []
    for a in reg["ridge_alphas"]:
        ridge_coefs.append(Ridge(alpha=a).fit(X_train, y_train).coef_)
    for a in reg["lasso_alphas"]:
        lasso_coefs.append(Lasso(alpha=a, max_iter=10000).fit(X_train, y_train).coef_)
    paths = {"Ridge": (reg["ridge_alphas"], ridge_coefs),
             "Lasso": (reg["lasso_alphas"], lasso_coefs)}
    fig = evaluation.plot_coefficient_paths(
        paths, current_alpha=alpha if model_name != "LinearRegression" else None)
    st.pyplot(fig)

# ---------------- Ensemble page ----------------
else:
    st.title("Part 2 — Ensemble Methods (breast_cancer)")
    st.markdown(
        "**Ensemble Methods**:\n"
        "- Use the `breast_cancer` dataset from `sklearn.datasets`.\n"
        "- Compare the performance (F1 Score and AUC) of `DecisionTreeClassifier`, `RandomForestClassifier` and `GradientBoostingClassifier`.\n"
        "- Tune the hyperparameters of each classifier using `GridSearchCV` with cross-validation.\n"
        "```python" \
        "from sklearn.datasets import load_breast_cancer\n" \
        "# Load the breast cancer dataset\n" \
        "breast_cancer = load_breast_cancer()\n" \
        "```")
    st.markdown("Pick a classifier and tweak its hyperparameters; F1, AUC and the ROC curve update live.")

    @st.cache_resource
    def get_clf_data():
        X_train, X_test, y_train, y_test, _, _ = data_loader.load_cancer_split()
        clf = training.precompute_clf_results(X_train, y_train)
        return X_train, X_test, y_train, y_test, clf

    X_train, X_test, y_train, y_test, clf = get_clf_data()

    model_name = st.sidebar.selectbox("Classifier", list(models.CLF_MODELS.keys()))
    best = clf[model_name].best_params_

    if model_name == "DecisionTree":
        max_depth = st.sidebar.slider("max_depth", 1, 20, int(best.get("max_depth") or 5))
        min_samples_split = st.sidebar.slider("min_samples_split", 2, 20, int(best.get("min_samples_split", 2)))
        model = DecisionTreeClassifier(
            max_depth=max_depth if max_depth != 20 else None,
            min_samples_split=min_samples_split, random_state=42)
    elif model_name == "RandomForest":
        n_estimators = st.sidebar.slider("n_estimators", 10, 300, int(best.get("n_estimators", 100)), step=10)
        max_depth = st.sidebar.slider("max_depth", 1, 30, int(best.get("max_depth") or 10))
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth != 30 else None,
            random_state=42, n_jobs=-1)
    else:
        n_estimators = st.sidebar.slider("n_estimators", 10, 300, int(best.get("n_estimators", 100)), step=10)
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5, float(best.get("learning_rate", 0.1)), step=0.01)
        max_depth = st.sidebar.slider("max_depth", 1, 10, int(best.get("max_depth", 3)))
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=42)

    with st.spinner("Fitting..."):
        model.fit(X_train, y_train)
        m = evaluation.classification_metrics(model, X_test, y_test)

    c1, c2 = st.columns(2)
    c1.metric("Test F1", f"{m['f1']:.3f}")
    c2.metric("Test AUC", f"{m['auc']:.3f}" if m['auc'] is not None else "n/a")

    if hasattr(model, "predict_proba"):
        roc_data = {model_name: (y_test, model.predict_proba(X_test)[:, 1])}
        fig = evaluation.plot_roc_curves(roc_data)
        st.pyplot(fig)
