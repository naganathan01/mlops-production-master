# 🐳 MLOps Docker Deployment Guide

## Prerequisites

- **Docker** 20.10+ installed
- **Docker Compose** 2.0+ installed
- **8GB+ RAM** recommended
- **10GB+ disk space** available

## Deployment Options

This guide covers three Docker deployment methods:
1. **Docker Compose (Recommended)** - Full stack with all services
2. **Individual Containers** - Step-by-step container deployment
3. **Development Setup** - For development and testing

---

## Option 1: Docker Compose Deployment (Recommended)

### Step 1: Prepare Environment

```bash
# Clone/download project
cd mlops-production

# Create required directories
mkdir -p {mlruns,data,logs,model-cache}

# Set permissions
chmod -R 755 mlruns data logs model-cache
```

### Step 2: Generate Sample Data

```bash
# Generate training data
python data/sample_training_data.py

# Or use Docker to generate data
docker run --rm -v $(pwd):/app -w /app python:3.9 \
  bash -c "pip install pandas numpy && python data/sample_training_data.py"
```

### Step 3: Configure Environment

```bash
# Copy and edit environment file
cp .env .env.local

# Edit .env.local with your settings
cat > .env.local << EOF
MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_NAME=production-model
MODEL_STAGE=Production
API_HOST=0.0.0.0
API_PORT=8080
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090
EOF
```

### Step 4: Start Full Stack

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 5: Train Initial Model

```bash
# Wait for MLflow to be ready
sleep 30

# Train model using Docker
docker-compose exec ml-api python src/model/train.py \
    --data-path data/sample_training_data.csv \
    --experiment-name "production-model" \
    --model-name "production-model" \
    --no-tuning
```

### Step 6: Promote Model

```bash
# Get run ID from MLflow UI (http://localhost:5000)
# Then promote model
RUN_ID="your-run-id-here"

docker-compose exec ml-api python src/utils/promote_model.py \
    --run-id $RUN_ID \
    --model-name "production-model" \
    --stage "Production"
```

### Step 7: Restart API to Load Model

```bash
# Restart API service to load promoted model
docker-compose restart ml-api

# Wait for restart
sleep 10
```

### Step 8: Test Deployment

```bash
# Health check
curl http://localhost:8080/health

# Model info
curl http://localhost:8080/model-info

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

### Access Points
- **MLflow UI**: http://localhost:5000
- **API Health**: http://localhost:8080/health
- **API Docs**: http://localhost:8080/docs
- **Metrics**: http://localhost:8080/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

---

## Option 2: Individual Container Deployment

### Step 1: Build Application Image

```bash
# Build the ML API image
docker build -t ml-model-api:latest .

# Verify image
docker images | grep ml-model-api
```

### Step 2: Create Docker Network

```bash
# Create network for services
docker network create mlops-network
```

### Step 3: Start MLflow Server

```bash
# Create MLflow data volume
docker volume create mlflow-data

# Start MLflow container
docker run -d \
  --name mlflow-server \
  --network mlops-network \
  -p 5000:5000 \
  -v mlflow-data:/mlflow \
  python:3.9-slim \
  bash -c "
    pip install mlflow==2.6.0 && \
    mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --default-artifact-root /mlflow/artifacts \
      --backend-store-uri sqlite:///mlflow/mlflow.db
  "

# Wait for MLflow to start
sleep 20

# Verify MLflow is running
curl http://localhost:5000/health
```

### Step 4: Train Model in Container

```bash
# Generate data (if not done already)
docker run --rm \
  -v $(pwd)/data:/app/data \
  python:3.9 \
  bash -c "
    pip install pandas numpy && \
    cd /app && \
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
df.to_csv('data/sample_training_data.csv', index=False)
print('Sample data created')
\"
  "

# Train model
docker run --rm \
  --network mlops-network \
  -v $(pwd):/app \
  -w /app \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  -e PYTHONPATH=/app \
  ml-model-api:latest \
  python src/model/train.py \
    --data-path data/sample_training_data.csv \
    --experiment-name "production-model" \
    --model-name "production-model" \
    --no-tuning
```

### Step 5: Get Run ID and Promote Model

```bash
# Get run ID
RUN_ID=$(docker run --rm \
  --network mlops-network \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  ml-model-api:latest \
  python -c "
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri('http://mlflow-server:5000')
client = MlflowClient()
experiment = client.get_experiment_by_name('production-model')
if experiment:
    runs = client.search_runs(experiment.experiment_id, order_by=['start_time DESC'], max_results=1)
    if runs:
        print(runs[0].info.run_id)
")

echo "Run ID: $RUN_ID"

# Promote model
docker run --rm \
  --network mlops-network \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  -v $(pwd):/app \
  -w /app \
  ml-model-api:latest \
  python src/utils/promote_model.py \
    --run-id $RUN_ID \
    --model-name "production-model" \
    --stage "Production"
```

### Step 6: Start API Container

```bash
# Start ML API container
docker run -d \
  --name ml-api \
  --network mlops-network \
  -p 8080:8080 \
  -p 9090:9090 \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  -e MODEL_NAME=production-model \
  -e MODEL_STAGE=Production \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model-cache:/app/model-cache \
  ml-model-api:latest

# Wait for API to start
sleep 15

# Check API health
curl http://localhost:8080/health
```

### Step 7: Test Individual Deployment

```bash
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

# View container logs
docker logs ml-api
docker logs mlflow-server
```

---

## Option 3: Development Setup

### Step 1: Development Compose

```bash
# Create development docker-compose override
cat > docker-compose.dev.yml << EOF
version: '3.8'
services:
  ml-api:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - /app/venv
    environment:
      - DEBUG=true
    command: tail -f /dev/null

  mlflow:
    volumes:
      - ./dev-mlruns:/mlflow/mlruns
EOF

# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Step 2: Development Workflow

```bash
# Enter development container
docker-compose exec ml-api bash

# Inside container - install development tools
pip install pytest black flake8 isort

# Run tests
pytest tests/

# Code formatting
black src/ tests/
flake8 src/ tests/

# Start API in development mode
python src/api/app.py
```

---

## 🔧 Configuration and Customization

### Environment Variables

```bash
# Core settings
MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_NAME=production-model
MODEL_STAGE=Production

# API settings
API_HOST=0.0.0.0
API_PORT=8080
LOG_LEVEL=INFO

# Performance settings
MAX_BATCH_SIZE=100
REQUEST_TIMEOUT=30

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

### Volume Mounts

```bash
# Data persistence
-v $(pwd)/data:/app/data

# Model cache
-v $(pwd)/model-cache:/app/model-cache

# Logs
-v $(pwd)/logs:/app/logs

# MLflow artifacts
-v mlflow-data:/mlflow
```

### Resource Limits

```yaml
# In docker-compose.yml
services:
  ml-api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

---

## 📊 Monitoring and Logs

### View Logs

```bash
# All services logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f ml-api
docker-compose logs -f mlflow

# Follow logs with timestamps
docker-compose logs -f -t ml-api
```

### Health Checks

```bash
# Check container health
docker-compose ps

# Service health endpoints
curl http://localhost:8080/health
curl http://localhost:5000/health

# Check resource usage
docker stats
```

### Monitoring Setup

```bash
# Access monitoring dashboards
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "MLflow: http://localhost:5000"
echo "API Docs: http://localhost:8080/docs"
```

---

## 🧪 Testing

### Run Tests in Containers

```bash
# Unit tests
docker-compose exec ml-api pytest tests/unit/ -v

# Integration tests
docker-compose exec ml-api python tests/integration/test_api_health.py \
  --endpoint http://localhost:8080

# Performance tests
docker-compose exec ml-api python tests/performance/load_test.py \
  --host http://localhost:8080 \
  --duration 30 \
  --users 5
```

### Load Testing

```bash
# External load test
docker run --rm --network host \
  ml-model-api:latest \
  python tests/performance/load_test.py \
    --host http://localhost:8080 \
    --duration 60 \
    --users 10 \
    --test-type mixed
```

---

## 🛑 Cleanup and Management

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop individual containers
docker stop ml-api mlflow-server
docker rm ml-api mlflow-server
```

### Cleanup Resources

```bash
# Remove images
docker rmi ml-model-api:latest

# Remove network
docker network rm mlops-network

# Remove volumes
docker volume rm mlflow-data

# Clean up all unused resources
docker system prune -a
```

### Backup Data

```bash
# Backup MLflow data
docker run --rm \
  -v mlflow-data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/mlflow-backup.tar.gz -C /data .

# Backup application data
tar czf backup/app-data-backup.tar.gz data/ model-cache/ logs/
```

---

## 🚨 Troubleshooting

### Common Issues

**Container Won't Start:**
```bash
# Check logs
docker-compose logs ml-api

# Check resource usage
docker stats

# Restart services
docker-compose restart ml-api
```

**Port Conflicts:**
```bash
# Check what's using ports
lsof -i :8080
lsof -i :5000

# Change ports in docker-compose.yml
ports:
  - "8081:8080"  # Change external port
```

**Model Loading Issues:**
```bash
# Check model registry
curl http://localhost:5000/api/2.0/mlflow/model-versions/search

# Restart API after model promotion
docker-compose restart ml-api
```

**Performance Issues:**
```bash
# Increase container resources
docker-compose down
# Edit docker-compose.yml to increase memory/CPU limits
docker-compose up -d

# Check container resources
docker stats
```

### Debug Mode

```bash
# Start with debug logging
docker-compose down
docker-compose up -d --environment LOG_LEVEL=DEBUG

# Access container shell
docker-compose exec ml-api bash

# Check environment variables
docker-compose exec ml-api env | grep ML
```

---

## 🎯 Production Considerations

### Security

```bash
# Use secrets for sensitive data
echo "your-secret-key" | docker secret create model_key -

# Non-root user in container
USER 1000:1000

# Read-only filesystem
--read-only --tmpfs /tmp
```

### Scaling

```bash
# Scale API replicas
docker-compose up -d --scale ml-api=3

# Load balancer configuration
# Add nginx or traefik for load balancing
```

### Backup Strategy

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker run --rm -v mlflow-data:/data -v $(pwd)/backup:/backup alpine \
  tar czf /backup/mlflow_$DATE.tar.gz -C /data .
```

---

🎉 **Congratulations!** Your MLOps system is now running in Docker with full containerization, monitoring, and production-ready features!
