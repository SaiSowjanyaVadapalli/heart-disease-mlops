"""
Shared inference helper. Loads the trained pipeline (preprocessing +
model, saved as a single joblib artifact by src/train.py) and exposes a
single `predict_one` function used by both the FastAPI service and the
unit tests, so "how we turn a request into a prediction" is defined once.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data_processing import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run `python src/train.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(features: dict[str, Any]) -> dict[str, Any]:
    """
    features: dict with the 13 raw clinical fields (see FEATURE_COLUMNS).
    Returns: {"prediction": 0/1, "label": str, "confidence": float}
    """
    missing = set(FEATURE_COLUMNS) - set(features.keys())
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    model = get_model()
    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])

    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    confidence = float(proba[pred])

    return {
        "prediction": pred,
        "label": "disease_present" if pred == 1 else "no_disease",
        "confidence": round(confidence, 4),
    }
