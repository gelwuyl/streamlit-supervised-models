"""Data loading, splitting, and scaling for the ML assignment.

WHY THESE CHOICES (important for the assignment):
- We split BEFORE scaling, and fit the scaler on the TRAIN set only, then
  transform both train and test. This avoids "data leakage" -- the test set
  must stay unseen when we compute statistics like mean/std.
- For the classifier we use stratify=y so the train/test split keeps the same
  proportion of malignant/benign cases as the full dataset.
- as_frame=True returns DataFrames with readable feature-name columns.
"""
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_diabetes_split(test_size=0.2, random_state=42):
    """Load the diabetes dataset, split, and scale (fit scaler on train only)."""
    try:
        diabetes = load_diabetes(as_frame=True)
        X = diabetes.data
        y = diabetes.target
        feature_names = list(X.columns)
        X = X.values
        y = y.values.ravel()
    except Exception as e:
        raise RuntimeError(f"Failed to load diabetes dataset: {e}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit on TRAIN only
    X_test = scaler.transform(X_test)         # transform TEST
    return X_train, X_test, y_train, y_test, scaler, feature_names


def load_cancer_split(test_size=0.2, random_state=42):
    """Load the breast cancer dataset, split (stratified), and scale."""
    try:
        cancer = load_breast_cancer(as_frame=True)
        X = cancer.data
        y = cancer.target
        feature_names = list(X.columns)
        X = X.values
        y = y.values.ravel()
    except Exception as e:
        raise RuntimeError(f"Failed to load breast cancer dataset: {e}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, feature_names
