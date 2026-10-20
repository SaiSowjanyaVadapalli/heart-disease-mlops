# Heart Disease Risk Prediction -- End-to-End MLOps Pipeline

AIMLCZG523 MLOps Assignment 1. A full pipeline -- data acquisition, EDA,
model training with MLflow tracking, automated testing, CI/CD, Docker
containerization, Kubernetes deployment, and monitoring -- for predicting
heart disease risk from the UCI Heart Disease (Cleveland) dataset.

## 1. Project layout

```
heart-disease-mlops/
├── scripts/download_data.py   # pulls + binarizes the UCI dataset
├── notebooks/01_eda.py        # EDA -> reports/figures/, reports/EDA_SUMMARY.md
├── src/
│   ├── data_processing.py     # cleaning + sklearn preprocessing pipeline
│   ├── train.py                # trains LR + RF, MLflow tracking, saves best model
│   └── predict.py              # shared inference helper (used by API + tests)
├── api/
│   ├── main.py                 # FastAPI app: /predict, /health, /metrics
│   └── schemas.py               # Pydantic request/response models
├── tests/                       # pytest unit + API integration tests
├── models/model.pkl             # trained pipeline artifact (generated, gitignored)
├── data/                        # raw + processed CSVs (generated, gitignored)
├── reports/                     # EDA figures, metrics.json (generated)
├── Dockerfile, docker-compose.yml
├── k8s/                         # Deployment + Service manifests for Minikube/Docker Desktop
├── monitoring/prometheus.yml
└── .github/workflows/ci.yml     # lint -> unit test -> train -> docker build/smoke-test
```

## 2. Setup (clean environment)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Data acquisition & EDA (Task 1)

```bash
python scripts/download_data.py
python notebooks/01_eda.py
```

`download_data.py` fetches `processed.cleveland.data` directly from the
**UCI Machine Learning Repository** (`archive.ics.uci.edu`), names the 14
clinical columns, converts `?` to `NaN`, and binarizes the multi-class
`num` target into `target` (0 = no disease, 1 = disease), per the
assignment's "binary target" requirement. If the UCI endpoint is
unreachable it falls back to a mirrored copy of the same Cleveland data.

`01_eda.py` produces, under `reports/figures/`:
- `class_balance.png` -- target distribution
- `feature_histograms.png` -- distributions of continuous features
- `correlation_heatmap.png`
- `target_vs_features.png` -- boxplots of age/thalach/oldpeak by diagnosis
- `chest_pain_vs_target.png`

and a written summary at `reports/EDA_SUMMARY.md`.

## 4. Feature engineering & model training (Tasks 2-4)

```bash
python src/train.py
```

This:
- builds a reusable `ColumnTransformer` (median-impute + scale numeric
  features; mode-impute + one-hot encode categorical ones) in
  `src/data_processing.py`,
- trains **Logistic Regression** and **Random Forest** inside a single
  `Pipeline` (preprocessing + estimator) using `GridSearchCV` with 5-fold
  stratified cross-validation, scored on ROC-AUC,
- evaluates both on a held-out 20% test split (accuracy, precision,
  recall, F1, ROC-AUC),
- logs every run (params, metrics, confusion matrix, ROC curve, the
  fitted model) to **MLflow**,
- saves the best-performing pipeline (by test ROC-AUC) to
  `models/model.pkl` with `joblib`, so the exact same object that does
  imputation + scaling + encoding + prediction is reused at inference
  time -- no train/serve skew.

Latest local run (reproducible via the seeded `random_state=42`):

| Model | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.902 | 0.885 | 0.839 | 0.929 | 0.881 | **0.966** |
| Random Forest | 0.902 | 0.852 | 0.806 | 0.893 | 0.847 | 0.948 |

**Selected model: Logistic Regression** (best test ROC-AUC). Full metrics
are in `reports/metrics.json`.

## 5. Experiment tracking with MLflow (Task 3)

Tracking store is a local SQLite DB (`mlflow.db`) rather than the plain
filesystem backend, since MLflow 3.x put the filesystem store into
maintenance mode. To browse runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 -- experiment "heart-disease-risk-classification"
has one run per model with logged params, metrics, and artifacts
(confusion matrix, ROC curve, the serialized model). Take a screenshot of
this UI for the report.

## 6. Unit tests (Task 5, part 1)

```bash
pytest tests/ -v --cov=src
```

- `tests/test_data_processing.py` -- offline, synthetic-data tests of
  cleaning + the preprocessing pipeline (duplicates, missing values,
  unseen categories).
- `tests/test_model.py` -- offline tests of pipeline training, predicted
  probabilities, and save/load round-tripping, using
  `sklearn.datasets.make_classification` (so this stage never needs
  internet access or a pre-trained model).
- `tests/test_api.py` -- integration tests against the live FastAPI app
  (skipped automatically if `models/model.pkl` doesn't exist yet).

## 7. CI/CD pipeline (Task 5, part 2)

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. **lint** -- `flake8` + `black --check` (fails the build on style/syntax errors)
2. **unit-tests** -- offline pytest suite with coverage, uploaded as an artifact
3. **train** -- downloads the dataset, runs EDA, trains + logs to MLflow,
   runs the API integration tests against the freshly trained model,
   uploads `models/model.pkl` and `reports/` as workflow artifacts
4. **docker-build-test** -- downloads the trained model artifact, builds
   the Docker image, runs the container, polls `/health`, and calls
   `/predict` with `sample_request.json` -- proving the model serves
   correctly in an isolated container, exactly as the assignment requires.

Each job fails loudly (non-zero exit, `docker logs` dumped on failure) so
broken code or a broken model never silently passes.

## 8. Model containerization (Task 6)

```bash
python src/train.py              # must run first -- bakes models/model.pkl into the image
docker build -t heart-disease-api:local .
docker run -p 8000:8000 heart-disease-api:local
```

Test it:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Expected response shape:
```json
{"prediction": 0, "label": "no_disease", "confidence": 0.8146, "model_name": "LogisticRegression"}
```

Interactive API docs: http://localhost:8000/docs

## 9. Production deployment to Kubernetes (Task 7)

See [`k8s/README.md`](k8s/README.md) for full step-by-step instructions
for both Docker Desktop's built-in Kubernetes and Minikube, including the
exact `kubectl` commands to verify pods/services and the screenshots to
capture for the report. Manifests: `k8s/deployment.yaml` (2 replicas,
liveness/readiness probes on `/health`, resource limits) and
`k8s/service.yaml` (`LoadBalancer`).

> This repository's generation environment has no Docker/kubectl
> available, so these steps were written and reviewed but must be
> executed and screenshotted on a machine with Docker Desktop
> (Kubernetes enabled) or Minikube installed.

## 10. Monitoring & logging (Task 8)

- **Request logging**: every API call is logged as a structured JSON line
  (method, path, status code, latency, and for `/predict`, the prediction
  + confidence) to both stdout (visible via `docker logs` / `kubectl
  logs`) and `logs/api.log`.
- **Metrics**: `GET /metrics` exposes Prometheus-format metrics
  (request counts, latency histograms, in-progress requests) via
  `prometheus-fastapi-instrumentator`.
- **Dashboard stack**: `docker-compose.yml` runs the API alongside
  Prometheus (scraping `/metrics` every 10s, config in
  `monitoring/prometheus.yml`) and Grafana:

  ```bash
  docker compose up --build
  ```
  - Prometheus UI: http://localhost:9090
  - Grafana: http://localhost:3000 (login `admin`/`admin`) -- add
    Prometheus (`http://prometheus:9090`) as a data source and build a
    panel on `http_requests_total` or `http_request_duration_seconds`.

## 11. Reproducing everything from a clean checkout

```bash
git clone <your-repo-url>
cd heart-disease-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python notebooks/01_eda.py
python src/train.py
pytest tests/ -v
docker build -t heart-disease-api:local .
docker run -p 8000:8000 heart-disease-api:local
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

## 12. Report & screenshots

The full written report (setup, EDA/modelling choices, experiment
tracking summary, architecture diagram, CI/CD and deployment workflow
screenshots) is in [`REPORT.md`](REPORT.md). Screenshots go in
`screenshots/` (see that folder's checklist).
