"""
Unit tests for src/data_processing.py. Deliberately offline and
deterministic (uses an in-memory DataFrame, not the downloaded dataset) so
this stage of the CI pipeline never depends on network access.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    clean_data,
    get_feature_target_split,
)


def _make_raw_df(
    n: int = 20, with_missing: bool = True, with_duplicates: bool = True
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {col: rng.normal(50, 10, n) for col in NUMERIC_FEATURES}
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.integers(0, 3, n)
    data[TARGET_COLUMN] = rng.integers(0, 2, n)
    df = pd.DataFrame(data)

    if with_missing:
        df.loc[0, "chol"] = np.nan
        df.loc[1, "thal"] = np.nan

    if with_duplicates:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    return df


def test_clean_data_drops_duplicates():
    df = _make_raw_df(with_duplicates=True)
    cleaned = clean_data(df)
    assert cleaned.duplicated().sum() == 0


def test_clean_data_keeps_only_expected_columns():
    df = _make_raw_df()
    df["extra_unused_column"] = 1
    cleaned = clean_data(df)
    assert set(cleaned.columns) == set(FEATURE_COLUMNS + [TARGET_COLUMN])


def test_clean_data_raises_on_missing_required_columns():
    df = _make_raw_df().drop(columns=["age"])
    with pytest.raises(Exception):
        clean_data(df)[FEATURE_COLUMNS]  # 'age' missing -> KeyError when selected


def test_get_feature_target_split_shapes():
    df = clean_data(_make_raw_df(with_duplicates=False))
    X, y = get_feature_target_split(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)
    assert y.dtype == int


def test_preprocessing_pipeline_handles_missing_values_and_scales():
    df = clean_data(_make_raw_df(with_missing=True, with_duplicates=False))
    X, _ = get_feature_target_split(df)

    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(X)

    # No NaNs should survive imputation
    assert not np.isnan(transformed).any()

    # Numeric columns should be roughly standardized (mean ~0) after scaling
    numeric_block = transformed[:, : len(NUMERIC_FEATURES)]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-6)


def test_preprocessing_pipeline_rejects_unknown_category_gracefully():
    df = clean_data(_make_raw_df(with_duplicates=False))
    X, _ = get_feature_target_split(df)

    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X)

    X_unseen = X.copy()
    X_unseen.loc[0, "cp"] = 999  # category never seen during fit
    # Should not raise thanks to handle_unknown="ignore"
    pipeline.transform(X_unseen)
