# Local Kubernetes Deployment (Minikube / Docker Desktop)

These steps run entirely on your own machine (Docker Desktop with Kubernetes
enabled, or Minikube) -- they need Docker and kubectl, which the sandbox used
to generate this project does not have.

## Option A: Docker Desktop Kubernetes

1. Enable Kubernetes in Docker Desktop (Settings -> Kubernetes -> Enable).
2. From the project root, train the model and build the image:
   ```
   python src/train.py
   docker build -t heart-disease-api:local .
   ```
3. Apply the manifests:
   ```
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```
4. Check status:
   ```
   kubectl get pods
   kubectl get svc heart-disease-api
   ```
   Docker Desktop's Kubernetes exposes `LoadBalancer` services on `localhost`
   directly, so once `EXTERNAL-IP` shows `localhost`, test with:
   ```
   curl http://localhost/health
   curl -X POST http://localhost/predict -H "Content-Type: application/json" -d @sample_request.json
   ```

## Option B: Minikube

1. Start Minikube and point your shell's Docker client at Minikube's daemon
   so the image you build is visible to the cluster:
   ```
   minikube start
   eval $(minikube docker-env)          # Windows PowerShell: & minikube -p minikube docker-env | Invoke-Expression
   python src/train.py
   docker build -t heart-disease-api:local .
   ```
2. Apply the manifests:
   ```
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```
3. Minikube doesn't provide a real cloud load balancer, so either run:
   ```
   minikube tunnel        # in a separate terminal, keep it running
   curl http://localhost/health
   ```
   or get a direct URL without tunneling:
   ```
   minikube service heart-disease-api --url
   ```

## Verifying the deployment (for your report screenshots)

```
kubectl get deployments
kubectl get pods -o wide
kubectl get svc heart-disease-api
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

Take screenshots of each of these, plus a successful `curl`/Postman call to
`/predict` and `/health`, and save them under `screenshots/`.

## Scaling and self-healing (nice optional demo)

```
kubectl scale deployment heart-disease-api --replicas=4
kubectl get pods -w
kubectl delete pod <one-pod-name>   # watch Kubernetes reschedule it automatically
```

## Cleanup

```
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
```
