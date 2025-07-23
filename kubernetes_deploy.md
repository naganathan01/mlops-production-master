# ☸️ MLOps Kubernetes Deployment Guide

## Prerequisites

- **Kubernetes cluster** (1.20+) with kubectl configured
- **Docker** for building images
- **Helm** 3.0+ (recommended for production)
- **Container registry** access (Docker Hub, ACR, ECR, etc.)
- **8GB+ cluster memory** recommended
- **LoadBalancer or Ingress controller** for external access

## Deployment Options

This guide covers:
1. **Local Kubernetes** (minikube/kind) - Development and testing
2. **Cloud Kubernetes** (AKS/EKS/GKE) - Production deployment
3. **Advanced Configuration** - Scaling, monitoring, and security

---

## Option 1: Local Kubernetes Deployment

### Step 1: Setup Local Cluster

#### Using Minikube
```bash
# Start minikube with sufficient resources
minikube start --memory=8192 --cpus=4 --driver=docker

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server

# Verify cluster
kubectl cluster-info
```

#### Using Kind
```bash
# Create kind cluster with ingress
cat > kind-config.yaml << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
- role: worker
- role: worker
EOF

kind create cluster --config kind-config.yaml --name mlops-cluster

# Install ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
```

### Step 2: Build and Push Images

```bash
# Build the ML API image
docker build -t ml-model-api:latest .

# Tag for local registry (minikube)
if using minikube:
  eval $(minikube docker-env)
  docker build -t ml-model-api:latest .

# For kind, load image directly
if using kind:
  kind load docker-image ml-model-api:latest --name mlops-cluster
```

### Step 3: Create Kubernetes Resources

```bash
# Apply namespace
kubectl apply -f infrastructure/kubernetes/namespace.yaml

# Create configmap for environment variables
kubectl create configmap ml-config \
  --from-literal=mlflow-uri=http://mlflow-service:5000 \
  --from-literal=model-name=production-model \
  --from-literal=log-level=INFO \
  -n ml-production

# Apply all manifests
kubectl apply -f infrastructure/kubernetes/
```

### Step 4: Deploy MLflow Server

```bash
# Create MLflow deployment
cat > mlflow-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: ml-production
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
      - name: mlflow
        image: python:3.9-slim
        ports:
        - containerPort: 5000
        command:
          - bash
          - -c
          - |
            pip install mlflow==2.6.0 &&
            mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root /mlflow/artifacts --backend-store-uri sqlite:///mlflow/mlflow.db
        volumeMounts:
        - name: mlflow-storage
          mountPath: /mlflow
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: mlflow-storage
        persistentVolumeClaim:
          claimName: mlflow-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
  namespace: ml-production
spec:
  selector:
    app: mlflow-server
  ports:
  - port: 5000
    targetPort: 5000
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-pvc
  namespace: ml-production
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

# Apply MLflow resources
kubectl apply -f mlflow-deployment.yaml
```

### Step 5: Wait for Services to Start

```bash
# Wait for MLflow to be ready
kubectl wait --for=condition=available --timeout=300s deployment/mlflow-server -n ml-production

# Check MLflow status
kubectl get pods -n ml-production -l app=mlflow-server

# Port forward to access MLflow UI
kubectl port-forward service/mlflow-service 5000:5000 -n ml-production &
```

### Step 6: Train Initial Model

```bash
# Generate sample data in a pod
kubectl run data-generator --rm -i --tty \
  --image=python:3.9 \
  --namespace=ml-production \
  --restart=Never -- bash -c "
pip install pandas numpy &&
python -c \"
import pandas as pd
import numpy as np
np.random.seed(42)
n_samples = 1000
data = {}
for i in range(10):
    data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
X = np.column_stack(list(data.values()))
target = (X[:, 0] + X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)
data['target'] = target
df = pd.DataFrame(data)
print(df.to_csv(index=False))
\" > /tmp/training_data.csv &&
cat /tmp/training_data.csv
" > training_data.csv

# Create training data ConfigMap
kubectl create configmap training-data \
  --from-file=training_data.csv \
  -n ml-production

# Train model using a job
cat > training-job.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training
  namespace: ml-production
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: ml-model-api:latest
        command:
          - python
          - src/model/train.py
          - --data-path
          - /data/training_data.csv
          - --experiment-name
          - production-model
          - --model-name
          - production-model
          - --no-tuning
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        - name: PYTHONPATH
          value: "/app"
        volumeMounts:
        - name: training-data
          mountPath: /data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: training-data
        configMap:
          name: training-data
      restartPolicy: Never
  backoffLimit: 3
EOF

kubectl apply -f training-job.yaml

# Monitor training job
kubectl logs -f job/model-training -n ml-production
```

### Step 7: Promote Model

```bash
# Get run ID from MLflow (access UI at http://localhost:5000)
# Create model promotion job
RUN_ID="your-run-id-here"

cat > promotion-job.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: model-promotion
  namespace: ml-production
spec:
  template:
    spec:
      containers:
      - name: promoter
        image: ml-model-api:latest
        command:
          - python
          - src/utils/promote_model.py
          - --run-id
          - $RUN_ID
          - --model-name
          - production-model
          - --stage
          - Production
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        - name: PYTHONPATH
          value: "/app"
      restartPolicy: Never
EOF

kubectl apply -f promotion-job.yaml
```

### Step 8: Deploy ML API

```bash
# Update deployment with correct image
kubectl apply -f infrastructure/kubernetes/deployment.yaml

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment/ml-api -n ml-production

# Check pod status
kubectl get pods -n ml-production -l app=ml-api
```

### Step 9: Expose Services

```bash
# Apply service and ingress
kubectl apply -f infrastructure/kubernetes/service.yaml
kubectl apply -f infrastructure/kubernetes/ingress.yaml

# For local clusters, get service URL
if using minikube:
  minikube service ml-api-service -n ml-production --url

if using kind:
  kubectl port-forward service/ml-api-service 8080:80 -n ml-production &
```

### Step 10: Test Local Deployment

```bash
# Health check
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature_0": 1.5, "feature_1": 2.3, "feature_2": -0.8,
      "feature_3": 0.1, "feature_4": -1.2, "feature_5": 0.7,
      "feature_6": 1.8, "feature_7": -0.3, "feature_8": 2.1,
      "feature_9": 0.9
    }
  }'
```

---

## Option 2: Cloud Kubernetes Deployment

### Azure Kubernetes Service (AKS)

#### Step 1: Setup AKS Cluster

```bash
# Login to Azure
az login

# Create resource group
az group create --name mlops-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group mlops-rg \
  --name mlops-aks \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group mlops-rg --name mlops-aks

# Install ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
```

#### Step 2: Setup Container Registry

```bash
# Create Azure Container Registry
az acr create \
  --resource-group mlops-rg \
  --name mlopsregistry \
  --sku Basic

# Attach ACR to AKS
az aks update \
  --name mlops-aks \
  --resource-group mlops-rg \
  --attach-acr mlopsregistry

# Build and push image
az acr build \
  --registry mlopsregistry \
  --image ml-model-api:latest \
  .
```

#### Step 3: Deploy to AKS

```bash
# Update deployment.yaml with ACR image
sed -i 's|ml-model-api:latest|mlopsregistry.azurecr.io/ml-model-api:latest|g' \
  infrastructure/kubernetes/deployment.yaml

# Deploy all resources
kubectl apply -f infrastructure/kubernetes/

# Get external IP
kubectl get ingress -n ml-production -w
```

### Amazon EKS

#### Step 1: Setup EKS Cluster

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name mlops-eks \
  --region us-west-2 \
  --nodegroup-name workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 6 \
  --managed

# Install ALB ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.6.0/docs/install/iam_policy.json

eksctl utils associate-iam-oidc-provider --region=us-west-2 --cluster=mlops-eks --approve
```

#### Step 2: Setup ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name ml-model-api --region us-west-2

# Get login token
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

# Build and push
docker build -t ml-model-api:latest .
docker tag ml-model-api:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/ml-model-api:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/ml-model-api:latest
```

### Google Kubernetes Engine (GKE)

#### Step 1: Setup GKE Cluster

```bash
# Set project
gcloud config set project <your-project-id>

# Create GKE cluster
gcloud container clusters create mlops-gke \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 6

# Get credentials
gcloud container clusters get-credentials mlops-gke --zone us-central1-a
```

#### Step 2: Setup Container Registry

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and push
docker build -t ml-model-api:latest .
docker tag ml-model-api:latest gcr.io/<project-id>/ml-model-api:latest
docker push gcr.io/<project-id>/ml-model-api:latest
```

---

## Option 3: Advanced Configuration

### Step 1: Helm Deployment

#### Create Helm Chart

```bash
# Create Helm chart
mkdir -p helm-chart/mlops
cd helm-chart/mlops

# Create Chart.yaml
cat > Chart.yaml << EOF
apiVersion: v2
name: mlops
description: MLOps Production Deployment
type: application
version: 1.0.0
appVersion: "1.0.0"
EOF

# Create values.yaml
cat > values.yaml << EOF
global:
  imageRegistry: ""
  imagePullSecrets: []

mlapi:
  image:
    repository: ml-model-api
    tag: latest
    pullPolicy: IfNotPresent
  
  replicaCount: 3
  
  resources:
    limits:
      cpu: 1000m
      memory: 2Gi
    requests:
      cpu: 500m
      memory: 1Gi
  
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80

  service:
    type: ClusterIP
    port: 80
    targetPort: 8080

mlflow:
  enabled: true
  persistence:
    enabled: true
    size: 20Gi
    storageClass: ""
  
  resources:
    limits:
      cpu: 500m
      memory: 1Gi
    requests:
      cpu: 250m
      memory: 512Mi

ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: mlapi.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mlapi-tls
      hosts:
        - mlapi.yourdomain.com

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: "admin123"
EOF
```

#### Create Templates

```bash
# Create templates directory
mkdir templates

# API Deployment template
cat > templates/deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "mlops.fullname" . }}-api
  labels:
    {{- include "mlops.labels" . | nindent 4 }}
    component: api
spec:
  {{- if not .Values.mlapi.autoscaling.enabled }}
  replicas: {{ .Values.mlapi.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "mlops.selectorLabels" . | nindent 6 }}
      component: api
  template:
    metadata:
      labels:
        {{- include "mlops.selectorLabels" . | nindent 8 }}
        component: api
    spec:
      containers:
      - name: api
        image: "{{ .Values.global.imageRegistry }}/{{ .Values.mlapi.image.repository }}:{{ .Values.mlapi.image.tag }}"
        imagePullPolicy: {{ .Values.mlapi.image.pullPolicy }}
        ports:
        - name: http
          containerPort: 8080
        - name: metrics
          containerPort: 9090
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://{{ include "mlops.fullname" . }}-mlflow:5000"
        - name: MODEL_NAME
          value: "production-model"
        - name: MODEL_STAGE
          value: "Production"
        resources:
          {{- toYaml .Values.mlapi.resources | nindent 10 }}
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
EOF
```

#### Deploy with Helm

```bash
# Install chart
helm install mlops . -n ml-production --create-namespace

# Upgrade deployment
helm upgrade mlops . -n ml-production

# Check status
helm status mlops -n ml-production
```

### Step 2: Advanced Monitoring

#### Prometheus Setup

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin123

# Port forward Grafana
kubectl port-forward service/prometheus-grafana 3000:80 -n monitoring &
```

#### Custom ServiceMonitor

```bash
cat > servicemonitor.yaml << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-api-metrics
  namespace: ml-production
spec:
  selector:
    matchLabels:
      app: ml-api
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF

kubectl apply -f servicemonitor.yaml
```

### Step 3: GitOps with ArgoCD

#### Install ArgoCD

```bash
# Create namespace
kubectl create namespace argocd

# Install ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Port forward ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443 &
```

#### Create Application

```bash
cat > argocd-app.yaml << EOF
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: mlops-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/mlops-repo
    targetRevision: HEAD
    path: helm-chart/mlops
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
EOF

kubectl apply -f argocd-app.yaml
```

---

## 🔧 Configuration Management

### Environment-Specific Configs

```bash
# Development values
cat > values-dev.yaml << EOF
mlapi:
  replicaCount: 1
  resources:
    limits:
      cpu: 500m
      memory: 1Gi
    requests:
      cpu: 250m
      memory: 512Mi
  autoscaling:
    enabled: false

ingress:
  hosts:
    - host: mlapi-dev.yourdomain.com
EOF

# Staging values
cat > values-staging.yaml << EOF
mlapi:
  replicaCount: 2
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 5

ingress:
  hosts:
    - host: mlapi-staging.yourdomain.com
EOF

# Production values
cat > values-prod.yaml << EOF
mlapi:
  replicaCount: 3
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20

ingress:
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "1000"
  hosts:
    - host: mlapi.yourdomain.com
EOF

# Deploy to different environments
helm install mlops-dev . -f values-dev.yaml -n ml-dev --create-namespace
helm install mlops-staging . -f values-staging.yaml -n ml-staging --create-namespace
helm install mlops-prod . -f values-prod.yaml -n ml-production --create-namespace
```

### Secrets Management

```bash
# Create secrets for sensitive data
kubectl create secret generic ml-secrets \
  --from-literal=database-password=mysecret \
  --from-literal=api-key=myapikey \
  -n ml-production

# Use External Secrets Operator (recommended)
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace

# AWS Secrets Manager example
cat > secret-store.yaml << EOF
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: ml-production
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
EOF
```

---

## 📊 Monitoring and Observability

### Logging with ELK Stack

```bash
# Install Elasticsearch
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch -n logging --create-namespace

# Install Kibana
helm install kibana elastic/kibana -n logging

# Install Filebeat
helm install filebeat elastic/filebeat -n logging
```

### Distributed Tracing

```bash
# Install Jaeger
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm install jaeger jaegertracing/jaeger -n tracing --create-namespace

# Configure application for tracing
# Add to deployment.yaml:
env:
- name: JAEGER_ENDPOINT
  value: "http://jaeger-collector:14268/api/traces"
```

### Custom Dashboards

```bash
# Import ML-specific Grafana dashboard
kubectl create configmap ml-dashboard \
  --from-file=infrastructure/monitoring/grafana-dashboard.json \
  -n monitoring

# Apply dashboard ConfigMap with labels
kubectl label configmap ml-dashboard grafana_dashboard=1 -n monitoring
```

---

## 🧪 Testing in Kubernetes

### Integration Testing

```bash
# Create test job
cat > integration-test-job.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: integration-tests
  namespace: ml-production
spec:
  template:
    spec:
      containers:
      - name: test-runner
        image: ml-model-api:latest
        command:
          - python
          - tests/integration/test_api_health.py
          - --endpoint
          - http://ml-api-service:80
        env:
        - name: PYTHONPATH
          value: "/app"
      restartPolicy: Never
EOF

kubectl apply -f integration-test-job.yaml
kubectl logs job/integration-tests -n ml-production
```

### Load Testing

```bash
# Create load test job
cat > load-test-job.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: load-test
  namespace: ml-production
spec:
  template:
    spec:
      containers:
      - name: load-tester
        image: ml-model-api:latest
        command:
          - python
          - tests/performance/load_test.py
          - --host
          - http://ml-api-service:80
          - --duration
          - "300"
          - --users
          - "50"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      restartPolicy: Never
EOF

kubectl apply -f load-test-job.yaml
```

---

## 🛑 Cleanup and Management

### Cleanup Commands

```bash
# Delete specific deployment
helm uninstall mlops -n ml-production

# Delete namespace and all resources
kubectl delete namespace ml-production

# Clean up local cluster
if using minikube:
  minikube delete

if using kind:
  kind delete cluster --name mlops-cluster

# Clean up cloud resources
if using AKS:
  az group delete --name mlops-rg --yes --no-wait

if using EKS:
  eksctl delete cluster --name mlops-eks

if using GKE:
  gcloud container clusters delete mlops-gke --zone us-central1-a
```

### Backup and Disaster Recovery

```bash
# Backup persistent volumes
kubectl get pv -o yaml > pv-backup.yaml

# Backup cluster state with Velero
velero backup create mlops-backup --include-namespaces ml-production

# Database backup (if using external DB)
kubectl exec -it mlflow-server-xxx -n ml-production -- \
  sqlite3 /mlflow/mlflow.db .dump > mlflow-backup.sql
```

---

## 🚨 Troubleshooting

### Common Issues

**Pod Stuck in Pending:**
```bash
# Check node resources
kubectl describe nodes

# Check pod events
kubectl describe pod <pod-name> -n ml-production

# Check resource quotas
kubectl get resourcequota -n ml-production
```

**Image Pull Errors:**
```bash
# Check image name and tag
kubectl describe pod <pod-name> -n ml-production

# Verify registry access
kubectl get secrets -n ml-production

# Test image pull manually
docker pull <image-name>
```

**Service Connection Issues:**
```bash
# Check service endpoints
kubectl get endpoints -n ml-production

# Test service connectivity
kubectl run debug --rm -i --tty --image=busybox --restart=Never -- nslookup ml-api-service.ml-production.svc.cluster.local

# Check network policies
kubectl get networkpolicies -n ml-production
```

**Performance Issues:**
```bash
# Check resource usage
kubectl top pods -n ml-production
kubectl top nodes

# Check HPA status
kubectl get hpa -n ml-production

# Scale manually if needed
kubectl scale deployment ml-api --replicas=5 -n ml-production
```

---

## 🎯 Production Best Practices

### Security

```bash
# Network policies
kubectl apply -f infrastructure/kubernetes/network-policies.yaml

# Pod security standards
kubectl label namespace ml-production pod-security.kubernetes.io/enforce=restricted

# RBAC
kubectl apply -f - << EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: ml-production
  name: ml-api-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ml-api-rolebinding
  namespace: ml-production
subjects:
- kind: ServiceAccount
  name: ml-api-sa
  namespace: ml-production
roleRef:
  kind: Role
  name: ml-api-role
  apiGroup: rbac.authorization.k8s.io
EOF
```

### High Availability

```bash
# Pod disruption budget
cat > pdb.yaml << EOF
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-api-pdb
  namespace: ml-production
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: ml-api
EOF

kubectl apply -f pdb.yaml

# Anti-affinity rules
# Add to deployment spec:
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - ml-api
        topologyKey: kubernetes.io/hostname
```

### Resource Management

```bash
# Resource quotas
cat > resource-quota.yaml << EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-production-quota
  namespace: ml-production
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    persistentvolumeclaims: "4"
    pods: "10"
EOF

kubectl apply -f resource-quota.yaml

# Limit ranges
cat > limit-range.yaml << EOF
apiVersion: v1
kind: LimitRange
metadata:
  name: ml-production-limits
  namespace: ml-production
spec:
  limits:
  - default:
      cpu: "500m"
      memory: "1Gi"
    defaultRequest:
      cpu: "250m"
      memory: "512Mi"
    type: Container
EOF

kubectl apply -f limit-range.yaml
```

---

🎉 **Congratulations!** Your MLOps system is now running on Kubernetes with enterprise-grade scalability, monitoring, and production-ready features!
