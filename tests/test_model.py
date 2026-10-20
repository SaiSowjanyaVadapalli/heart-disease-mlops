"""
Unit tests for model-building code. Uses sklearn's make_classification to
stay fast and offline -- these tests check that our Pipeline wiring,
training, and persistence round-trip correctly, not the UCI dataset's
actual accuracy (that's validated separately via src/train.py + MLflow).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_processing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessing_pipeline,
)


def _synthetic_patient_frame(n: int = 300, random_state: int = 0):
    X_raw, y = make_classification(
        n_samples=n,
        n_features=len(NUMERIC_FEATURES),
        n_informative=len(NUMERIC_FEATURES),
        n_redundant=0,
        class_sep=2.0,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(X_raw, columns=NUMERIC_FEATURES)
    for col in CATEGORICAL_FEATURES:
        df[col] = rng.integers(0, 3, n)
    return df, pd.Series(y)


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            ("model", LogisticRegression(max_iter=1000, random_state=0)),
        ]
    )


def test_pipeline_trains_and_beats_random_baseline():
    X, y = _synthetic_patient_frame()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))

    assert acc > 0.6  # meaningfully better than a 50% random-guess baseline


def test_pipeline_predict_proba_returns_valid_probabilities():
    X, y = _synthetic_patient_frame()
    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X.head(5))
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_pipeline_save_load_roundtrip_is_deterministic(tmp_path):
    X, y = _synthetic_patient_frame()
    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)

    original_preds = pipeline.predict(X)
    reloaded_preds = reloaded.predict(X)
    assert np.array_equal(original_preds, reloaded_preds)
