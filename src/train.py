"""
Train, cross-validate, and track Logistic Regression and Random Forest
classifiers for heart disease risk prediction, with full MLflow experiment
tracking (params, metrics, plots, and the fitted pipeline as an artifact).

The best model (by cross-validated ROC-AUC) is re-fit on the full training
split and saved to models/model.pkl as a single sklearn Pipeline object
(preprocessing + estimator) so inference code never has to reconstruct
feature engineering by hand.

Usage:
    python src/train.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    RocCurveDisplay,
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_processing import (  # noqa: E402
    build_preprocessing_pipeline,
    clean_data,
    get_feature_target_split,
    load_raw_data,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "metrics.json"
MLFLOW_DB_PATH = ROOT / "mlflow.db"

RANDOM_STATE = 42

MODEL_SPECS = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
        },
    },
}


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("heart-disease-risk-classification")

    df = clean_data(load_raw_data(RAW_PATH))
    X, y = get_feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train.assign(target=y_train).to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_test.assign(target=y_test).to_csv(PROCESSED_DIR / "test.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    best_name, best_score, best_pipeline = None, -1.0, None

    for name, spec in MODEL_SPECS.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessing_pipeline()),
                    ("model", spec["estimator"]),
                ]
            )

            grid = GridSearchCV(
                pipeline, spec["param_grid"], cv=cv, scoring="roc_auc", n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_est = grid.best_estimator_

            metrics = evaluate(best_est, X_test, y_test)
            results[name] = {
                "best_params": grid.best_params_,
                "cv_best_roc_auc": grid.best_score_,
                "test_metrics": metrics,
            }

            mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
            mlflow.log_metric("cv_roc_auc", grid.best_score_)
            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            # Confusion matrix artifact
            fig, ax = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay.from_estimator(
                best_est, X_test, y_test, ax=ax, cmap="Blues"
            )
            ax.set_title(f"Confusion Matrix - {name}")
            cm_path = FIG_DIR / f"confusion_matrix_{name}.png"
            fig.tight_layout()
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))

            # ROC curve artifact
            fig, ax = plt.subplots(figsize=(5, 5))
            RocCurveDisplay.from_estimator(best_est, X_test, y_test, ax=ax)
            ax.set_title(f"ROC Curve - {name}")
            roc_path = FIG_DIR / f"roc_curve_{name}.png"
            fig.tight_layout()
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(roc_path))

            mlflow.sklearn.log_model(
                best_est, name=name, serialization_format="cloudpickle"
            )

            print(f"[{name}] cv_roc_auc={grid.best_score_:.4f} test_metrics={metrics}")

            if metrics["roc_auc"] > best_score:
                best_name, best_score, best_pipeline = (
                    name,
                    metrics["roc_auc"],
                    best_est,
                )

    joblib.dump(best_pipeline, MODELS_DIR / "model.pkl")
    results["selected_model"] = best_name
    METRICS_PATH.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nSelected best model: {best_name} (test ROC-AUC={best_score:.4f})")
    print(f"Saved pipeline -> {MODELS_DIR / 'model.pkl'}")
    print(f"Saved metrics  -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
