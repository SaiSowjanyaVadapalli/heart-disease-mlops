"""
Data acquisition script for the Heart Disease UCI dataset.

Downloads the official "Heart Disease" dataset (Cleveland subset) directly
from the UCI Machine Learning Repository, parses the raw space-separated
file (which has no header and uses '?' for missing values), assigns the
standard 14 clinical feature names, and binarizes the multi-class target
('num': 0-4) into a binary label ('target': 0 = no disease, 1 = disease),
exactly as required by the assignment ("binary target").

Usage:
    python scripts/download_data.py
"""

import io
import sys
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
# Fallback mirror used only if the UCI endpoint is unreachable (e.g. network
# restrictions in a CI runner). Same Cleveland data, pre-parsed with headers.
FALLBACK_CSV_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
)

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_FILE = RAW_DIR / "heart_disease.csv"


def _fetch_from_uci() -> pd.DataFrame:
    req = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open("processed.cleveland.data") as f:
            df = pd.read_csv(f, header=None, names=COLUMNS, na_values="?")

    # Binarize target: 0 = no disease, 1-4 = presence of disease -> 1
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df


def _fetch_from_fallback() -> pd.DataFrame:
    req = urllib.request.Request(
        FALLBACK_CSV_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        df = pd.read_csv(resp)
    # This mirror already ships a binary 'target' column with the same schema.
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading dataset from UCI ML Repository: {UCI_ZIP_URL}")
        df = _fetch_from_uci()
        source = "UCI Machine Learning Repository (processed.cleveland.data)"
    except Exception as exc:  # noqa: BLE001
        print(
            f"UCI download failed ({exc}); falling back to mirror: {FALLBACK_CSV_URL}"
        )
        df = _fetch_from_fallback()
        source = "GitHub mirror of UCI Cleveland Heart Disease dataset"

    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {OUT_FILE}")
    print(f"Source: {source}")
    print(f"Missing values per column:\n{df.isna().sum()}")
    print(f"Target class balance:\n{df['target'].value_counts(normalize=True)}")


if __name__ == "__main__":
    sys.exit(main())
