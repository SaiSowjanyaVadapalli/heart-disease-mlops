# Pipeline Walkthrough Video -- Narration Script

Target length: ~6-8 minutes. Record your screen (OBS, Windows Game Bar
`Win+G`, or Zoom's local recording) while reading/paraphrasing this.
Each section lists what to have on screen and what to say. Cut anything
that runs long -- the assignment asks for a *short* recording, not a
full re-derivation of the report.

Before recording: run `python src/train.py` once so `models/model.pkl`
exists, and have a terminal + browser tabs ready (VS Code, MLflow UI,
GitHub Actions, Docker Desktop, browser on `localhost:8000/docs`).

---

## 0. Intro (15-20 sec)

**Screen:** Project folder open in VS Code / file explorer, README.md visible.

**Say:**
> "Hi, this is a walkthrough of my MLOps Assignment 1 pipeline -- a heart
> disease risk classifier built on the UCI Heart Disease dataset, with
> data processing, model training and tracking, automated testing,
> CI/CD, containerization, Kubernetes deployment, and monitoring."

---

## 1. Data acquisition & EDA (45-60 sec)

**Screen:** Terminal, then `reports/figures/` images.

**Say:**
> "The dataset comes straight from the UCI Machine Learning Repository
> -- this script downloads the raw Cleveland file, names the clinical
> columns, handles the '?' missing values, and binarizes the target
> into disease present or absent."

**Do:** run (or show already-run output of):
```
python scripts/download_data.py
python notebooks/01_eda.py
```

**Say, while flipping through 2-3 figures:**
> "EDA shows a fairly balanced target class, some skew in cholesterol
> and ST depression, and the features most correlated with disease --
> thalassemia result, number of major vessels, and exercise-induced
> angina -- all clinically sensible."

---

## 2. Feature engineering & model training (60-75 sec)

**Screen:** `src/data_processing.py` briefly, then terminal running `src/train.py`.

**Say:**
> "Feature engineering lives in one reusable sklearn ColumnTransformer --
> numeric features are median-imputed and scaled, categorical features
> are mode-imputed and one-hot encoded. I train two models, Logistic
> Regression and Random Forest, each tuned with grid search and 5-fold
> stratified cross-validation, scored on ROC-AUC."

**Do:** run `python src/train.py` (or show it already run) and let the
printed per-model metrics appear on screen.

**Say:**
> "Logistic Regression comes out ahead on the held-out test set -- about
> 88.5% accuracy and 0.97 ROC-AUC -- so that's the model that gets
> saved and deployed."

---

## 3. Experiment tracking with MLflow (45-60 sec)

**Screen:** Browser at `http://127.0.0.1:5000` (run `mlflow ui --backend-store-uri sqlite:///mlflow.db` beforehand).

**Say:**
> "Every run is tracked in MLflow -- parameters, cross-validated and
> test metrics, and artifacts like the confusion matrix, ROC curve, and
> the serialized model itself."

**Do:** click into one run, show the metrics panel and the artifacts tab.

---

## 4. Unit tests & CI/CD (60-75 sec)

**Screen:** Terminal running pytest, then GitHub Actions tab in the browser.

**Say:**
> "There are 14 automated tests covering data cleaning, the
> preprocessing pipeline, model training, and the API -- all offline and
> fast, so they don't depend on network access."

**Do:** run `pytest tests/ -v` and let it finish green.

**Say, switching to GitHub Actions:**
> "On every push, GitHub Actions runs four stages: lint, unit tests,
> model training with the real dataset, and finally a Docker build that
> spins up the container and curls the live API to prove it serves
> correctly before anything is considered done."

**Do:** scroll through a completed (green) workflow run, briefly opening
the docker-build-test job's logs to show the curl response.

---

## 5. Containerization (45-60 sec)

**Screen:** Terminal.

**Say:**
> "The API is a FastAPI app wrapping the trained pipeline, with a
> `/predict` endpoint, a `/health` check, and Prometheus metrics at
> `/metrics`."

**Do:**
```
docker build -t heart-disease-api:local .
docker run -p 8000:8000 heart-disease-api:local
```
Then in another terminal or browser:
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```
Or just open `http://localhost:8000/docs` and run `/predict` from the
Swagger UI.

**Say:**
> "And it builds and runs cleanly as an isolated container, returning a
> prediction and a confidence score."

---

## 6. Kubernetes deployment (60-75 sec)

**Screen:** Terminal with kubectl.

**Say:**
> "For production deployment, I wrote a Kubernetes Deployment with two
> replicas and liveness/readiness probes, plus a LoadBalancer Service."

**Do:**
```
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc heart-disease-api
curl http://localhost/health
```

**Say:**
> "Kubernetes is keeping two replicas healthy, and the service is
> reachable from outside the cluster."

(Optional, if time allows) show `kubectl delete pod <name>` and the
replacement pod coming up automatically.

---

## 7. Monitoring (30-45 sec)

**Screen:** Prometheus targets page, then Grafana.

**Say:**
> "Every request is logged as structured JSON, and Prometheus scrapes
> the API's `/metrics` endpoint for request counts and latency, which
> feeds this Grafana dashboard."

**Do:** `docker compose up` (if not already running), show Prometheus
"Targets" page with the API marked UP, then a Grafana panel.

---

## 8. Wrap-up (15-20 sec)

**Screen:** README.md or the project folder tree.

**Say:**
> "That's the full pipeline -- from raw data to a monitored, containerized,
> Kubernetes-deployed API, with experiment tracking and CI/CD tying it
> all together. Code and docs are in the linked GitHub repository."

---

## Recording checklist

- [ ] Audio levels checked (a quiet room, no background noise)
- [ ] Screen resolution readable when re-watched (zoom terminal font up)
- [ ] All 8 sections covered, total under ~8 minutes
- [ ] Export as .mp4, name it something like `heart-disease-mlops-demo.mp4`
- [ ] Upload / link it per your course's submission instructions
