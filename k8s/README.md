# Local Kubernetes Deployment (Minikube / Docker Desktop / Codespaces)

These steps run entirely on your own machine (or in the cloud, see
Option D) -- they need Docker and/or kubectl, which the sandbox used to
generate this project does not have.

> **On a corporate-managed machine?** Docker Desktop requires a paid
> subscription at large companies and may be blocked by IT policy, and
> enabling Hyper-V (Option C) touches a system-level Windows feature your
> IT department may want to know about. The safest option with **zero
> changes to your laptop** is **Option D: GitHub Codespaces** -- it runs
> entirely in the cloud, with nothing installed locally.

## Option D: GitHub Codespaces (recommended for corporate machines -- no local install)

Runs Docker and a full Kubernetes cluster inside a free, browser-based
cloud dev environment. Nothing is installed on your laptop; no admin
rights, no reboot, no licensing concerns.

1. Push this repository to GitHub (see the main README).
2. On the repo's GitHub page: **Code** button -> **Codespaces** tab ->
   **Create codespace on main**. Wait ~1-2 minutes for it to build --
   `.devcontainer/devcontainer.json` automatically provisions Docker
   (via the docker-in-docker feature) and kubectl/Minikube/Helm.
3. In the Codespace's terminal (this is a full Linux VM in your browser):
   ```bash
   pip install -r requirements.txt     # usually already done by postCreateCommand
   python scripts/download_data.py
   python src/train.py
   docker build -t heart-disease-api:local .
   docker run -d -p 8000:8000 --name heart-api heart-disease-api:local
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
   docker rm -f heart-api
   ```
4. Start a Kubernetes cluster and deploy:
   ```bash
   minikube start --driver=docker
   minikube image load heart-disease-api:local
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl get pods
   kubectl get svc heart-disease-api
   ```
5. Reach the service with a simple port-forward (works the same
   everywhere, no tunnel/LoadBalancer quirks):
   ```bash
   kubectl port-forward svc/heart-disease-api 8080:80
   ```
   Then, in a second terminal tab:
   ```bash
   curl http://localhost:8080/health
   curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @sample_request.json
   ```
   VS Code (in the browser) will also show a notification offering to
   open port 8080 -- click it, or use the **Ports** tab, to open
   `/docs` directly in a new browser tab for a live demo / recording.
6. For the monitoring stack: `docker compose up -d` and port-forward
   9090 (Prometheus) and 3000 (Grafana) the same way, or just use the
   **Ports** tab's forwarded links.
7. Take your screenshots and record your video directly from this
   browser-based VS Code session -- it's a real terminal running real
   `docker`/`kubectl` commands, which is all the assignment requires.
8. When done, delete the Codespace (Codespaces tab on GitHub) to stop
   using your free quota. Free tier: 120 core-hours/month, far more than
   this assignment needs.

## Option A: Docker Desktop Kubernetes

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

## Option C: Minikube + Hyper-V (no Docker Desktop)

Uses Windows' built-in Hyper-V feature as the VM backend and builds the
image *inside* Minikube itself (`minikube image build`), so no Docker
Engine or Docker Desktop is ever installed on the host.

### C.1 Enable Hyper-V (one-time, requires admin + a reboot)

In an **elevated** PowerShell (right-click PowerShell -> "Run as
Administrator"):
```powershell
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
Restart-Computer
```

### C.2 Install Minikube and kubectl

After reboot, in a normal PowerShell:
```powershell
winget install minikube
winget install Kubernetes.kubectl
```
(or `choco install minikube kubernetes-cli -y` if you prefer Chocolatey)

Close and reopen your terminal so the new PATH entries are picked up, then
confirm:
```powershell
minikube version
kubectl version --client
```

### C.3 Create a Hyper-V virtual switch (one-time)

Still in an **elevated** PowerShell:
```powershell
Get-NetAdapter                      # note the Name of your active Wi-Fi/Ethernet adapter
New-VMSwitch -Name "minikube-switch" -NetAdapterName "<your-adapter-name>" -AllowManagementOS $true
```

### C.4 Start Minikube

Minikube's Hyper-V driver needs admin rights, so keep using an
**elevated** PowerShell for every `minikube` command below:
```powershell
minikube start --driver=hyperv --hyperv-virtual-switch="minikube-switch"
```

### C.5 Build the image inside Minikube (no local Docker needed)

```powershell
cd "path\to\heart-disease-mlops"
python src\train.py                              # must run first -- produces models\model.pkl
minikube image build -t heart-disease-api:local .
```

### C.6 Deploy and verify

```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc heart-disease-api
```

Minikube doesn't provide a real cloud load balancer, so either:
```powershell
minikube tunnel        # separate elevated terminal, keep it running
curl http://localhost/health
```
or get a direct URL without tunneling:
```powershell
minikube service heart-disease-api --url
```

### C.7 Cleanup

```powershell
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
minikube stop
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
