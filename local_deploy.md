# 🚀 MLOps Local Deployment Guide

## Prerequisites

Before starting, ensure you have:
- **Python 3.9+** installed
- **Git** for version control
- **8GB+ RAM** recommended
- **5GB+ disk space** available

## Step 1: Project Setup

### 1.1 Clone/Download Project
```bash
# If using Git
git clone <your-mlops-repo>
cd mlops-production

# Or download and extract the project files
```

### 1.2 Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 1.3 Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# On Windows:
# set PYTHONPATH=%PYTHONPATH%;%cd%
```

## Step 2: Generate Sample Data

### 2.1 Create Training Data
```bash
# Generate sample training data
python data/sample_training_data.py
```

### 2.2 Validate Data Quality
```bash
# Run data validation
python src/data/validate_data.py \
    --data-path data/sample_training_data.csv \
    --output data_validation_report.json

# Check validation results
cat data_validation_report.json
```

## Step 3: Start MLflow Server

### 3.1 Launch MLflow
```bash
# Start MLflow server in background
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root ./mlruns \
    --backend-store-uri sqlite:///mlflow.db &

# Save the process ID
MLFLOW_PID=$!
echo "MLflow PID: $MLFLOW_PID"
```

### 3.2 Verify MLflow is Running
```bash
# Wait for MLflow to start
sleep 15

# Test MLflow connectivity
curl http://localhost:5000/health

# Access MLflow UI: http://localhost:5000
```

## Step 4: Train Your Model

### 4.1 Set Environment Variables
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MODEL_NAME=production-model
```

### 4.2 Train Model
```bash
# Train model with hyperparameter tuning
python src/model/train.py \
    --data-path data/sample_training_data.csv \
    --experiment-name "production-model" \
    --model-name "production-model"

# For faster training (skip hyperparameter tuning):
python src/model/train.py \
    --data-path data/sample_training_data.csv \
    --experiment-name "production-model" \
    --model-name "production-model" \
    --no-tuning
```

### 4.3 Get Run ID
```bash
# Check MLflow UI at http://localhost:5000
# Copy the Run ID from the latest experiment run
# Or use this command to get the latest run ID:

python -c "
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('http://localhost:5000')
client = MlflowClient()

experiment = client.get_experiment_by_name('production-model')
if experiment:
    runs = client.search_runs(experiment.experiment_id, order_by=['start_time DESC'], max_results=1)
    if runs:
        print('Latest Run ID:', runs[0].info.run_id)
    else:
        print('No runs found')
else:
    print('Experiment not found')
"
```

## Step 5: Evaluate Model

### 5.1 Run Model Evaluation
```bash
# Replace RUN_ID with your actual run ID
RUN_ID="your-run-id-here"

python src/model/evaluate.py \
    --model-uri "runs:/$RUN_ID/model" \
    --test-data data/sample_training_data.csv \
    --output-report model_evaluation.json

# Check evaluation results
cat model_evaluation.json
```

## Step 6: Promote Model to Production

### 6.1 Promote Model
```bash
# Replace RUN_ID with your actual run ID
python src/utils/promote_model.py \
    --run-id $RUN_ID \
    --model-name "production-model" \
    --stage "Production" \
    --description "Initial production deployment"
```

### 6.2 Verify Model Promotion
```bash
# Check in MLflow UI that model is in Production stage
# Navigate to Models tab at http://localhost:5000
```

## Step 7: Start API Server

### 7.1 Configure API
```bash
# Set API environment variables
export MODEL_NAME=production-model
export MODEL_STAGE=Production
export MLFLOW_TRACKING_URI=http://localhost:5000
export API_HOST=0.0.0.0
export API_PORT=8080
export LOG_LEVEL=INFO
```

### 7.2 Launch API Server
```bash
# Start API server
python src/api/app.py

# API will start on http://localhost:8080
# Keep this terminal open (API runs in foreground)
```

## Step 8: Test Your Deployment

### 8.1 Health Check
```bash
# In a new terminal, test health endpoint
curl http://localhost:8080/health

# Expected response:
# {"status":"healthy","model_loaded":true,"uptime":...}
```

### 8.2 Model Info
```bash
# Get model information
curl http://localhost:8080/model-info

# Check API documentation
# Open browser: http://localhost:8080/docs
```

### 8.3 Make Predictions
```bash
# Single prediction
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

# Batch prediction
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
```

### 8.4 Check Metrics
```bash
# View Prometheus metrics
curl http://localhost:8080/metrics

# View model statistics
curl http://localhost:8080/stats
```

## Step 9: Run Integration Tests

### 9.1 Basic API Tests
```bash
# Run API health tests
python tests/integration/test_api_health.py \
    --endpoint http://localhost:8080 \
    --max-retries 3
```

### 9.2 Performance Testing
```bash
# Run load test
python tests/performance/load_test.py \
    --host http://localhost:8080 \
    --duration 30 \
    --users 5 \
    --test-type single
```

### 9.3 Complete System Test
```bash
# Run full end-to-end test (in a new terminal)
chmod +x test_mlops_complete.sh
./test_mlops_complete.sh
```

## Step 10: Monitoring and Management

### 10.1 Access Points
- **MLflow UI**: http://localhost:5000
- **API Health**: http://localhost:8080/health
- **API Documentation**: http://localhost:8080/docs
- **Metrics Endpoint**: http://localhost:8080/metrics
- **Model Statistics**: http://localhost:8080/stats

### 10.2 Logs Location
```bash
# API logs (if running in background)
tail -f api.log

# MLflow logs
tail -f mlflow.log

# Application logs are structured JSON
```

## 🛑 Cleanup and Shutdown

### Stop Services
```bash
# Stop API server (if running in background)
pkill -f "python src/api/app.py"

# Stop MLflow server
pkill -f "mlflow server"

# Or use specific PID
kill $MLFLOW_PID
```

### Cleanup Files
```bash
# Remove temporary files (optional)
rm -rf mlruns/
rm mlflow.db
rm *.log

# Deactivate virtual environment
deactivate
```

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Check what's using the port
lsof -i :5000  # MLflow
lsof -i :8080  # API

# Kill processes using the port
sudo kill -9 $(lsof -t -i:5000)
```

**MLflow Connection Issues:**
```bash
# Verify MLflow is running
curl http://localhost:5000/health

# Check MLflow logs
tail -f mlflow.log
```

**Model Not Loading:**
```bash
# Check if model exists in registry
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
print(mlflow.search_model_versions('name=\"production-model\"'))
"
```

**API Issues:**
```bash
# Check API logs
tail -f api.log

# Test with verbose curl
curl -v http://localhost:8080/health
```

### Performance Optimization

**For Better Performance:**
```bash
# Use more workers (if needed)
uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --workers 4

# Increase model cache
export MODEL_CACHE_SIZE=1024
```

## 🎯 Next Steps

1. **Explore MLflow UI** to understand experiment tracking
2. **Try different data** by modifying `data/sample_training_data.py`
3. **Experiment with model parameters** in `src/model/train.py`
4. **Set up monitoring** by configuring Prometheus/Grafana
5. **Scale up** by moving to Docker or Kubernetes deployment

## 📚 Additional Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Project Structure**: See `README.md` for detailed architecture
- **API Schema**: Visit http://localhost:8080/docs when API is running

---

🎉 **Congratulations!** Your MLOps system is now running locally with full model training, serving, and monitoring capabilities!
