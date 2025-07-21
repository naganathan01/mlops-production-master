#!/bin/bash
# setup.sh - Complete MLOps setup script

echo "🚀 Setting up MLOps Production Environment..."

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p {src/{api,model,data,utils},tests/{unit,integration,performance},infrastructure/{kubernetes,monitoring},data}

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
echo "⚙️  Setting environment variables..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MODEL_NAME=production-model

# Generate sample data
echo "📊 Generating sample training data..."
python3 -c "
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

# Start MLflow server in background
echo "🔄 Starting MLflow server..."
mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns --backend-store-uri sqlite:///mlflow.db &
MLFLOW_PID=$!
echo "MLflow PID: $MLFLOW_PID"

# Wait for MLflow to start
echo "⏳ Waiting for MLflow server to start..."
sleep 10

# Train initial model
echo "🎯 Training initial model..."
python src/model/train.py \
    --data-path data/sample_training_data.csv \
    --experiment-name "production-model-setup" \
    --model-name "production-model"

# Get the run ID (you'll need to extract this from the output)
echo "📝 Check MLflow UI at http://localhost:5000 to get the run ID"
echo "Then run: python src/utils/promote_model.py --run-id YOUR_RUN_ID --model-name production-model --stage Production"

echo "✅ Setup complete!"
echo "🔗 MLflow UI: http://localhost:5000"
echo "📖 Next steps:"
echo "  1. Get run ID from MLflow UI"
echo "  2. Promote model: python src/utils/promote_model.py --run-id RUN_ID --model-name production-model --stage Production"  
echo "  3. Start API: python src/api/app.py"
echo "  4. Test API: curl http://localhost:8080/health"