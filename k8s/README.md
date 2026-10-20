# Local Kubernetes Deployment (Minikube / Docker Desktop)

These steps run entirely on your own machine -- they need Docker and/or
kubectl, which the sandbox used to generate this project does not have.

> **On a corporate-managed machine?** Docker Desktop requires a paid
> subscription at large companies and may be blocked by IT policy. Use
> **Option C** below instead -- it uses only Windows' built-in Hyper-V
> feature and the free, open-source Minikube/kubectl, with no Docker
> Desktop install at all.

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
