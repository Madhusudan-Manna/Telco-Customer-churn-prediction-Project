"""
preprocessing.py
----------------
Reusable preprocessing utilities for the Telco Customer Churn project.
Mirrors the logic used in notebook/03_preprocessing.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV and fix TotalCharges dtype.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with TotalCharges as float and NaNs filled with 0.
    """
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


# ---------------------------------------------------------------------------
# 2. Basic Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the customerID column and encode the Churn target as 0/1.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (output of load_data).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe without customerID; Churn is now int (0 or 1).
    """
    df = df.copy()
    df = df.drop("customerID", axis=1)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------

def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'tenure_group' categorical column based on the tenure (months).

    Bins:
        0-12   → short-term
        13-24  → medium
        25-48  → long-term
        49-72  → loyal

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with the new tenure_group column appended.
    """
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    )
    return df


# ---------------------------------------------------------------------------
# 4. Train / Test Split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Separate features from the target and create a stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of clean_data).
    target : str
        Name of the target column.
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ---------------------------------------------------------------------------
# 5. Preprocessing Pipeline (ColumnTransformer)
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_features: list,
    categorical_features: list
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numeric features and one-hot-encodes
    categorical features (including the engineered tenure_group).

    Parameters
    ----------
    numeric_features : list
        List of numeric column names.
    categorical_features : list
        List of categorical column names (include 'tenure_group' if added).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted preprocessor ready for fit_transform / transform.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    return preprocessor


# ---------------------------------------------------------------------------
# 6. Simple encode-and-scale helper (used in notebook cell for quick scaling)
# ---------------------------------------------------------------------------

def encode_and_scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    One-hot-encode all categorical columns with pd.get_dummies, align the
    test set to match train columns, then apply StandardScaler.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test  : pd.DataFrame

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    scaler                        : fitted StandardScaler instance
    """
    X_train_enc = pd.get_dummies(X_train)
    X_test_enc  = pd.get_dummies(X_test)

    # Align so test has the same columns as train (fill unseen with 0)
    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_enc, join="left", axis=1, fill_value=0
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled  = scaler.transform(X_test_enc)

    return X_train_scaled, X_test_scaled, scaler
