# Screenshots to capture (run on a machine with Docker Desktop / Minikube)

Save each as a PNG in this folder, named as below, then insert into REPORT.docx
at the matching `[Insert screenshot...]` placeholder.

- [ ] `01_mlflow_ui.png` -- MLflow UI showing both runs + metrics
- [ ] `02_mlflow_run_artifacts.png` -- one run's artifact view (confusion matrix, ROC curve, model files)
- [ ] `03_pytest_passing.png` -- `pytest tests/ -v` terminal output, all green
- [ ] `04_github_actions_run.png` -- a green GitHub Actions run with all 4 jobs passing
- [ ] `05_docker_build.png` -- `docker build -t heart-disease-api:local .` succeeding
- [ ] `06_docker_run_curl.png` -- `docker run` + `curl /predict` response
- [ ] `07_kubectl_get_pods.png` -- `kubectl get pods` showing 2/2 running
- [ ] `08_kubectl_get_svc.png` -- `kubectl get svc heart-disease-api` showing EXTERNAL-IP
- [ ] `09_k8s_curl_predict.png` -- successful curl/Postman call against the cluster endpoint
- [ ] `10_prometheus_targets.png` -- Prometheus "Targets" page showing the API as UP
- [ ] `11_grafana_dashboard.png` -- Grafana panel built on the Prometheus data source
