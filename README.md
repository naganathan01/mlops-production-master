# Complete MLOps Production Implementation

## 🎯 What This Project Solves

This comprehensive MLOps implementation addresses critical machine learning production challenges:

### **Core Problems Solved:**

1. **Model Deployment Complexity** - Automated, scalable model serving with monitoring
2. **Data Quality Issues** - Automated validation and drift detection
3. **Model Performance Degradation** - Continuous monitoring and alerting
4. **Lack of Reproducibility** - Version control for models, data, and code
5. **Manual Operations** - End-to-end automation from training to deployment
6. **Scalability Challenges** - Kubernetes-based auto-scaling deployment
7. **Compliance & Governance** - Audit trails and model governance

## 🚀 Complete Setup & Testing Guide

### **Step 1: Environment Setup**

```bash
# Clone and setup the project
git clone <your-repo>
cd mlops-production

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### **Step 2: Generate Sample Data & Train Model**

```bash
# Run the complete setup script
chmod +x setup.sh
./setup.sh

# OR run steps manually:

# 1. Generate sample training data
python -c "
import pandas as pd
import numpy as np
np.random.seed(42)
n_samples = 1000
n_features = 10
data = {}
for i in range(n_features):
    data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
X = np.column_stack(list(data.values()))
target = (X[:, 0] + X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)
data['target'] = target
df = pd.DataFrame(data)
df.to_csv('data/sample_training_data.csv', index=False)
print('✅ Sample data created!')
"

# 2. Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns --backend-store-uri sqlite:///mlflow.db &
sleep 10

# 3. Train the model
python src/model/train.py \
    --data-path data/sample_training_data.csv \
    --experiment-name "production-model" \
    --model-name "production-model"
```

### **Step 3: Validate Data Quality**

```bash
# Run data validation
python src/data/validate_data.py \
    --data-path data/sample_training_data.csv \
    --output data_validation_report.json

# Check validation results
cat data_validation_report.json
```

### **Step 4: Promote Model to Production**

```bash
# Get the run ID from MLflow UI (http://localhost:5000)
# Then promote the model
RUN_ID="your-run-id-here"  # Replace with actual run ID

python src/utils/promote_model.py \
    --run-id $RUN_ID \
    --model-name "production-model" \
    --stage "Production" \
    --description "Initial production deployment"
```

### **Step 5: Start the API Server**

```bash
# Set environment variables
export MODEL_NAME=production-model
export MODEL_STAGE=Production
export MLFLOW_TRACKING_URI=http://localhost:5000

# Start the API
python src/api/app.py

# The API will be available at http://localhost:8080
```

### **Step 6: Test the Complete System**

```bash
# In another terminal, run integration tests
python tests/integration/test_api_health.py --endpoint http://localhost:8080

# Test individual endpoints
curl http://localhost:8080/health
curl http://localhost:8080/model-info

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature_0": 1.5,
      "feature_1": 2.3,
      "feature_2": -0.8,
      "feature_3": 0.1,
      "feature_4": -1.2,
      "feature_5": 0.7,
      "feature_6": 1.8,
      "feature_7": -0.3,
      "feature_8": 2.1,
      "feature_9": 0.9
    }
  }'

# Test batch prediction
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "features": {
          "feature_0": 1.5, "feature_1": 2.3, "feature_2": -0.8,
          "feature_3": 0.1, "feature_4": -1.2, "feature_5": 0.7,
          "feature_6": 1.8, "feature_7": -0.3, "feature_8": 2.1,
          "feature_9": 0.9
        }
      },
      {
        "features": {
          "feature_0": -1.5, "feature_1": -2.3, "feature_2": 0.8,
          "feature_3": -0.1, "feature_4": 1.2, "feature_5": -0.7,
          "feature_6": -1.8, "feature_7": 0.3, "feature_8": -2.1,
          "feature_9": -0.9
        }
      }
    ]
  }'

# Check metrics
curl http://localhost:8080/metrics
curl http://localhost:8080/stats
```

## 🔄 Complete MLOps Flow Demonstration

### **Flow 1: Data-Driven Model Updates**

```bash
# 1. New data arrives - simulate with new dataset
python -c "
import pandas as pd
import numpy as np
np.random.seed(123)  # Different seed for new data
n_samples = 500
data = {}
for i in range(10):
    data[f'feature_{i}'] = np.random.normal(0.2, 1.2, n_samples)  # Slightly different distribution
X = np.column_stack(list(data.values()))
target = (X[:, 0] + X[:, 1] + 0.3 * X[:, 2] > 0).astype(int)
data['target'] = target
df = pd.DataFrame(data)
df.to_csv('data/new_training_data.csv', index=False)
print('📊 New training data created')
"

# 2. Validate new data quality
python src/data/validate_data.py \
    --data-path data/new_training_data.csv \
    --output new_data_validation.json

# 3. Retrain model with new data
python src/model/train.py \
    --data-path data/new_training_data.csv \
    --experiment-name "production-model-retrain" \
    --model-name "production-model-v2"

# 4. Evaluate new model
NEW_RUN_ID="get-from-mlflow-ui"
python src/model/evaluate.py \
    --model-uri "runs:/$NEW_RUN_ID/model" \
    --test-data data/sample_training_data.csv \
    --output-report model_evaluation.json

# 5. If evaluation passes, promote new model
python src/utils/promote_model.py \
    --run-id $NEW_RUN_ID \
    --model-name "production-model-v2" \
    --stage "Staging"

# 6. API automatically picks up new model (or restart API)
curl -X POST http://localhost:8080/reload-model
```

### **Flow 2: Monitoring & Drift Detection**

```bash
# 1. Generate predictions to build baseline
for i in {1..100}; do
  curl -s -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d "{
      \"features\": {
        \"feature_0\": $(python -c "import random; print(random.gauss(0, 1))"),
        \"feature_1\": $(python -c "import random; print(random.gauss(0, 1))"),
        \"feature_2\": $(python -c "import random; print(random.gauss(0, 1))"),
        \"feature_3\": $(python -c "import random; print(random.gauss(0, 1))"),
        \"feature_4\": $(python -c "import random; print(random.gauss(0, 1))"),
        \"feature_5\": $(python -c "import random; print(random.gauss(0, 1))"),
        \"feature_6\": $(python -c "import random; print(random.gauss(0, 1))"),
        \"feature_7\": $(python -c "import random; print(random.gauss(0, 1))"),
        \"feature_8\": $(python -c "import random; print(random.gauss(0, 1))"),
        \"feature_9\": $(python -c "import random; print(random.gauss(0, 1))")
      }
    }" > /dev/null
done

# 2. Check monitoring stats
curl http://localhost:8080/stats

# 3. View Prometheus metrics
curl http://localhost:8080/metrics | grep ml_
```

## 🐳 Docker Deployment

### **Option 1: Docker Compose (Full Stack)**

```bash
# Start entire stack (MLflow + API + Monitoring)
docker-compose up -d

# Wait for services to start
sleep 30

# Test the dockerized API
curl http://localhost:8080/health

# Access UIs:
# - MLflow: http://localhost:5000
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### **Option 2: Individual Docker Build**

```bash
# Build the API image
docker build -t ml-model-api:latest .

# Run the container
docker run -d \
  --name ml-api \
  -p 8080:8080 \
  -e MODEL_NAME=production-model \
  -e MODEL_STAGE=Production \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  ml-model-api:latest

# Test
curl http://localhost:8080/health
```

## ☸️ Kubernetes Deployment

### **Option 1: Local Kubernetes (minikube/kind)**

```bash
# Start local cluster
minikube start
# OR
kind create cluster --name ml-cluster

# Apply manifests
kubectl apply -f infrastructure/kubernetes/

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment/ml-api -n ml-production

# Port forward to test
kubectl port-forward service/ml-api-service 8080:80 -n ml-production

# Test
curl http://localhost:8080/health
```

### **Option 2: Cloud Deployment (Azure/AWS/GCP)**

```bash
# Configure your cloud CLI (az/aws/gcloud)
# Update image registry in deployment.yaml
# Apply with proper ingress configuration

kubectl apply -f infrastructure/kubernetes/
kubectl get ingress -n ml-production
```

## 🔍 Testing & Validation Scenarios

### **Scenario 1: Model Performance Regression**

```bash
# Simulate poor model by using wrong features
python -c "
import requests
import json

# Send predictions with wrong feature patterns
for i in range(50):
    response = requests.post('http://localhost:8080/predict', json={
        'features': {f'feature_{j}': 999 if j < 5 else 0 for j in range(10)}
    })
    print(f'Prediction {i}: {response.json().get(\"prediction\", \"Error\")}')
"

# Check if drift/anomaly is detected
curl http://localhost:8080/stats
```

### **Scenario 2: Load Testing**

```bash
# Simple load test
python -c "
import requests
import threading
import time

def make_requests():
    for i in range(100):
        try:
            response = requests.post('http://localhost:8080/predict', 
                json={'features': {f'feature_{j}': j*0.1 for j in range(10)}},
                timeout=5
            )
            print(f'Request {i}: {response.status_code}')
        except Exception as e:
            print(f'Request {i}: Error - {e}')
        time.sleep(0.1)

# Start 5 concurrent threads
threads = []
for _ in range(5):
    t = threading.Thread(target=make_requests)
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print('Load test completed')
"
```

### **Scenario 3: Failure Recovery**

```bash
# Test API resilience
# 1. Stop API
pkill -f "python src/api/app.py"

# 2. Try requests (should fail)
curl http://localhost:8080/health

# 3. Restart API
python src/api/app.py &
sleep 10

# 4. Test recovery
curl http://localhost:8080/health
```

## 📊 Monitoring & Observability

### **Key Metrics to Monitor:**

1. **Model Performance Metrics:**
   - Prediction latency (95th percentile < 100ms)
   - Throughput (requests/second)
   - Error rate (< 1%)
   - Model accuracy (> 70%)

2. **Business Metrics:**
   - Conversion rate from predictions
   - Revenue impact
   - User satisfaction scores

3. **Infrastructure Metrics:**
   - CPU/Memory usage
   - Network latency
   - Disk I/O

4. **Data Quality Metrics:**
   - Feature drift scores
   - Data distribution changes
   - Missing value rates

### **Alert Conditions:**

```bash
# Example alert queries (Prometheus format)
# High latency: histogram_quantile(0.95, ml_prediction_latency_seconds) > 0.1
# Low accuracy: ml_model_accuracy < 0.7
# High drift: ml_drift_score > 2.0
# High error rate: rate(ml_predictions_total{status="error"}[5m]) > 0.01
```

## 🔧 Troubleshooting Guide

### **Common Issues & Solutions:**

1. **Model Loading Errors:**
   ```bash
   # Check MLflow connection
   curl http://localhost:5000/health
   
   # Verify model exists
   python -c "import mlflow; print(mlflow.search_model_versions('name=\"production-model\"'))"
   ```

2. **API Connection Issues:**
   ```bash
   # Check if port is available
   netstat -tulpn | grep 8080
   
   # Check logs
   docker logs ml-api  # if using Docker
   ```

3. **Data Validation Failures:**
   ```bash
   # Check data format
   head -5 data/sample_training_data.csv
   
   # Run validation with debug
   python src/data/validate_data.py --data-path data/sample_training_data.csv --output debug.json
   ```

## 📈 Performance Optimization Tips

1. **Model Optimization:**
   - Use model quantization for faster inference
   - Implement model caching
   - Consider ensemble methods for better accuracy

2. **API Optimization:**
   - Enable gzip compression
   - Implement connection pooling
   - Use async processing for batch requests

3. **Infrastructure Optimization:**
   - Use GPU for inference if needed
   - Implement horizontal pod autoscaling
   - Configure proper resource limits

## 🔄 CI/CD Pipeline Testing

### **Test the CI Pipeline Locally:**

```bash
# Simulate CI steps
pytest tests/unit/ -v
black --check src/ tests/
flake8 src/ tests/

# Build and test Docker image
docker build -t test-ml-api .
docker run -d --name test-api -p 8081:8080 test-ml-api
sleep 15
python tests/integration/test_api_health.py --endpoint http://localhost:8081
docker stop test-api && docker rm test-api
```

This comprehensive MLOps implementation provides a complete production-ready machine learning system with monitoring, automation, and best practices for scalable ML operations.