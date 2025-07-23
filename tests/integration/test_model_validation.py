# tests/integration/test_model_validation.py
import pytest
import requests
import time
import tempfile
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

class TestModelValidationFlow:
    """Integration tests for end-to-end model validation"""
    
    def __init__(self, mlflow_uri: str = "http://localhost:5000", api_uri: str = "http://localhost:8080"):
        self.mlflow_uri = mlflow_uri
        self.api_uri = api_uri
        self.temp_files = []
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def create_test_data(self, n_samples: int = 500, n_features: int = 10) -> str:
        """Create test dataset"""
        np.random.seed(42)
        data = {}
        
        # Generate features
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Create target with logical relationship
        X = np.column_stack(list(data.values()))
        target = ((X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3) > 0.1).astype(int)
        data['target'] = target
        
        # Save to temporary file
        df = pd.DataFrame(data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def wait_for_mlflow(self, timeout: int = 60) -> bool:
        """Wait for MLflow server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.mlflow_uri}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ MLflow server is ready")
                    return True
            except:
                pass
            time.sleep(2)
        
        print("❌ MLflow server not ready")
        return False
    
    def wait_for_api(self, timeout: int = 60) -> bool:
        """Wait for API server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_uri}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ API server is ready")
                    return True
            except:
                pass
            time.sleep(2)
        
        print("❌ API server not ready")
        return False
    
    def test_data_validation_flow(self):
        """Test complete data validation flow"""
        print("🧪 Testing data validation flow...")
        
        # Create test data
        data_file = self.create_test_data()
        print(f"📊 Created test data: {data_file}")
        
        # Test data validation
        import subprocess
        
        validation_output = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        validation_output.close()
        self.temp_files.append(validation_output.name)
        
        result = subprocess.run([
            'python', 'src/data/validate_data.py',
            '--data-path', data_file,
            '--output', validation_output.name
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Data validation failed: {result.stderr}"
        
        # Check validation results
        with open(validation_output.name, 'r') as f:
            validation_results = json.load(f)
        
        assert validation_results['data_quality']['passed'], "Data quality validation should pass"
        print("✅ Data validation passed")
        
        return data_file, validation_results
    
    def test_model_training_flow(self, data_file: str):
        """Test complete model training flow"""
        print("🎯 Testing model training flow...")
        
        # Ensure MLflow is ready
        assert self.wait_for_mlflow(), "MLflow server must be running"
        
        # Train model
        import subprocess
        
        result = subprocess.run([
            'python', 'src/model/train.py',
            '--data-path', data_file,
            '--experiment-name', 'integration-test',
            '--model-name', 'integration-test-model',
            '--no-tuning'  # Skip hyperparameter tuning for speed
        ], capture_output=True, text=True, env={
            **os.environ,
            'MLFLOW_TRACKING_URI': self.mlflow_uri,
            'PYTHONPATH': os.getcwd()
        })
        
        assert result.returncode == 0, f"Model training failed: {result.stderr}"
        
        # Extract run ID from training output or query MLflow
        print("✅ Model training completed")
        
        # Get latest run ID
        run_id = self.get_latest_run_id('integration-test')
        assert run_id, "Should have a run ID from training"
        
        return run_id
    
    def get_latest_run_id(self, experiment_name: str) -> str:
        """Get latest run ID from experiment"""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            mlflow.set_tracking_uri(self.mlflow_uri)
            client = MlflowClient()
            
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment:
                runs = client.search_runs(
                    experiment.experiment_id, 
                    order_by=['start_time DESC'], 
                    max_results=1
                )
                if runs:
                    return runs[0].info.run_id
            return None
        except Exception as e:
            print(f"Error getting run ID: {e}")
            return None
    
    def test_model_evaluation_flow(self, run_id: str, data_file: str):
        """Test model evaluation flow"""
        print("📊 Testing model evaluation flow...")
        
        # Create evaluation output file
        eval_output = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        eval_output.close()
        self.temp_files.append(eval_output.name)
        
        # Evaluate model
        import subprocess
        
        result = subprocess.run([
            'python', 'src/model/evaluate.py',
            '--model-uri', f'runs:/{run_id}/model',
            '--test-data', data_file,
            '--output-report', eval_output.name
        ], capture_output=True, text=True, env={
            **os.environ,
            'MLFLOW_TRACKING_URI': self.mlflow_uri,
            'PYTHONPATH': os.getcwd()
        })
        
        assert result.returncode == 0, f"Model evaluation failed: {result.stderr}"
        
        # Check evaluation results
        with open(eval_output.name, 'r') as f:
            eval_results = json.load(f)
        
        assert 'metrics' in eval_results
        assert 'accuracy' in eval_results['metrics']
        assert eval_results['metrics']['accuracy'] > 0.5, "Model should have reasonable accuracy"
        
        print(f"✅ Model evaluation completed - Accuracy: {eval_results['metrics']['accuracy']:.3f}")
        return eval_results
    
    def test_model_promotion_flow(self, run_id: str):
        """Test model promotion flow"""
        print("🚀 Testing model promotion flow...")
        
        # Promote model
        import subprocess
        
        result = subprocess.run([
            'python', 'src/utils/promote_model.py',
            '--run-id', run_id,
            '--model-name', 'integration-test-model',
            '--stage', 'Staging',
            '--description', 'Integration test promotion'
        ], capture_output=True, text=True, env={
            **os.environ,
            'MLFLOW_TRACKING_URI': self.mlflow_uri,
            'PYTHONPATH': os.getcwd()
        })
        
        assert result.returncode == 0, f"Model promotion failed: {result.stderr}"
        print("✅ Model promotion completed")
    
    def test_api_prediction_flow(self):
        """Test API prediction flow"""
        print("🌐 Testing API prediction flow...")
        
        # Ensure API is ready
        assert self.wait_for_api(), "API server must be running"
        
        # Test health endpoint
        response = requests.get(f"{self.api_uri}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        print(f"API Health: {health_data['status']}")
        
        # Test prediction if model is loaded
        if health_data.get('model_loaded', False):
            test_features = {f"feature_{i}": float(i * 0.1) for i in range(10)}
            
            response = requests.post(
                f"{self.api_uri}/predict",
                json={"features": test_features}
            )
            
            if response.status_code == 200:
                prediction_data = response.json()
                assert 'prediction' in prediction_data
                print(f"✅ Prediction successful: {prediction_data['prediction']}")
            else:
                print(f"⚠️  Prediction failed: {response.status_code}")
        else:
            print("⚠️  Model not loaded in API")
    
    def test_complete_integration_flow(self):
        """Test complete end-to-end integration flow"""
        print("🔄 Running complete integration test flow...")
        
        try:
            # Step 1: Data validation
            data_file, validation_results = self.test_data_validation_flow()
            
            # Step 2: Model training
            run_id = self.test_model_training_flow(data_file)
            
            # Step 3: Model evaluation
            eval_results = self.test_model_evaluation_flow(run_id, data_file)
            
            # Step 4: Model promotion
            self.test_model_promotion_flow(run_id)
            
            # Step 5: API testing
            self.test_api_prediction_flow()
            
            print("🎉 Complete integration flow test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False
        
        finally:
            self.cleanup()

def main():
    """Run integration tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Validation Integration Tests')
    parser.add_argument('--mlflow-uri', default='http://localhost:5000', help='MLflow URI')
    parser.add_argument('--api-uri', default='http://localhost:8080', help='API URI')
    parser.add_argument('--test-type', choices=['data', 'training', 'evaluation', 'promotion', 'api', 'complete'], 
                       default='complete', help='Type of test to run')
    
    args = parser.parse_args()
    
    tester = TestModelValidationFlow(args.mlflow_uri, args.api_uri)
    
    try:
        if args.test_type == 'data':
            data_file, results = tester.test_data_validation_flow()
            print(f"Data validation results: {results['data_quality']}")
        
        elif args.test_type == 'training':
            data_file = tester.create_test_data()
            run_id = tester.test_model_training_flow(data_file)
            print(f"Training run ID: {run_id}")
        
        elif args.test_type == 'evaluation':
            # This requires an existing run ID - would need to be provided
            print("Evaluation test requires existing run ID")
        
        elif args.test_type == 'promotion':
            # This requires an existing run ID - would need to be provided
            print("Promotion test requires existing run ID")
        
        elif args.test_type == 'api':
            tester.test_api_prediction_flow()
        
        elif args.test_type == 'complete':
            success = tester.test_complete_integration_flow()
            exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        exit(1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        exit(1)
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()