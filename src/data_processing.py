"""
Data cleaning and feature-engineering pipeline for the Heart Disease dataset.

Keeping this logic in a single module (rather than inline in a notebook)
means the exact same transformations are used during training, during
unit tests, and during inference in the API -- this is what guarantees
reproducibility (task 4) and gives us something concrete to unit test
(task 5).
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "target"

# Continuous clinical measurements -> median-impute + scale
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Coded / categorical clinical attributes -> mode-impute + one-hot encode
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the CSV produced by scripts/download_data.py."""
    df = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing expected columns: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning shared by training and EDA:
      - drop exact duplicate rows
      - restrict to the columns the model actually uses
      - coerce dtypes (the UCI file loads 'ca'/'thal' as float because of NaNs)

    Missing-value imputation itself is deliberately left to the sklearn
    Pipeline (build_preprocessing_pipeline) rather than done here, so that
    the imputer is *fit on the training split only* and reused verbatim at
    inference time -- avoiding train/serve skew and data leakage.
    """
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    ColumnTransformer that imputes + scales numeric features and imputes +
    one-hot encodes categorical ones. Returned unfitted; callers (train.py,
    tests) wrap it in a sklearn Pipeline alongside an estimator so the
    whole thing is fit/saved/loaded as one reproducible artifact.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
