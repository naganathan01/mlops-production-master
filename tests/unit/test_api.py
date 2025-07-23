# tests/unit/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np

from src.api.app import app

@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)

@pytest.fixture
def mock_model_service():
    """Mock ML model service"""
    with patch('src.api.app.ml_service') as mock:
        mock.is_healthy.return_value = True
        mock.model_version = "test-model:v1.0"
        mock.get_uptime.return_value = 100.0
        mock.predict.return_value = {
            'prediction': 0.85,
            'confidence': 0.92,
            'model_version': 'test-model:v1.0',
            'timestamp': '2024-01-01T00:00:00',
            'latency_ms': 25.0
        }
        yield mock

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_healthy(self, client, mock_model_service):
        """Test health check when service is healthy"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True
        assert 'uptime' in data
        assert 'version' in data

    def test_health_check_unhealthy(self, client):
        """Test health check when service is unhealthy"""
        with patch('src.api.app.ml_service') as mock:
            mock.is_healthy.return_value = False
            mock.model = None
            mock.get_uptime.return_value = 50.0
            mock.model_version = None
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data['status'] == 'unhealthy'
            assert data['model_loaded'] is False

class TestPredictionEndpoint:
    """Test prediction endpoints"""
    
    def test_single_prediction_success(self, client, mock_model_service):
        """Test successful single prediction"""
        payload = {
            "features": {
                f"feature_{i}": float(i * 0.1) 
                for i in range(10)
            }
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert 'prediction' in data
        assert 'model_version' in data
        assert 'timestamp' in data
        assert data['prediction'] == 0.85

    def test_single_prediction_model_not_loaded(self, client):
        """Test prediction when model is not loaded"""
        with patch('src.api.app.ml_service') as mock:
            mock.is_healthy.return_value = False
            
            payload = {
                "features": {f"feature_{i}": 0.5 for i in range(10)}
            }
            
            response = client.post("/predict", json=payload)
            assert response.status_code == 503

    def test_batch_prediction_success(self, client, mock_model_service):
        """Test successful batch prediction"""
        payload = {
            "requests": [
                {
                    "features": {f"feature_{i}": 0.1 * i for i in range(10)}
                },
                {
                    "features": {f"feature_{i}": 0.2 * i for i in range(10)}
                }
            ]
        }
        
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert 'results' in data
        assert 'total' in data
        assert 'successful' in data
        assert data['total'] == 2
        assert data['successful'] == 2

    def test_prediction_invalid_features(self, client, mock_model_service):
        """Test prediction with invalid feature format"""
        payload = {
            "features": "invalid"  # Should be dict
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    def test_model_info_success(self, client, mock_model_service):
        """Test successful model info retrieval"""
        mock_model_service.model_version = "test-model:v1.0"
        mock_model_service.model_uri = "models:/test-model/Production"
        mock_model_service.loaded_at = "2024-01-01T00:00:00"
        mock_model_service.feature_names = [f"feature_{i}" for i in range(10)]
        
        response = client.get("/model-info")
        assert response.status_code == 200
        
        data = response.json()
        assert data['model_version'] == "test-model:v1.0"
        assert data['model_uri'] == "models:/test-model/Production"
        assert len(data['features']) == 10

    def test_model_info_not_loaded(self, client):
        """Test model info when model is not loaded"""
        with patch('src.api.app.ml_service') as mock:
            mock.is_healthy.return_value = False
            
            response = client.get("/model-info")
            assert response.status_code == 503

class TestFeedbackEndpoint:
    """Test feedback endpoint"""
    
    def test_record_feedback_success(self, client, mock_model_service):
        """Test successful feedback recording"""
        response = client.post(
            "/feedback",
            params={
                "prediction_id": "test-123",
                "actual_outcome": 1.0,
                "user_satisfaction": 4.5
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'feedback recorded'

class TestStatsEndpoint:
    """Test statistics endpoint"""
    
    def test_get_stats_success(self, client, mock_model_service):
        """Test successful statistics retrieval"""
        mock_model_service.model_metrics.get_summary_stats.return_value = {
            'total_predictions': 100,
            'avg_prediction': 0.5,
            'drift_score': 1.2
        }
        mock_model_service.business_metrics.get_business_summary.return_value = {
            'conversion_rate': 0.15,
            'avg_satisfaction': 4.2
        }
        
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert 'model_metrics' in data
        assert 'business_metrics' in data
        assert 'service_uptime' in data

# tests/unit/test_model_training.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.model.train import ModelTrainer

@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 100
    data = {}
    for i in range(5):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    X = np.column_stack(list(data.values()))
    target = (X[:, 0] + X[:, 1] > 0).astype(int)
    data['target'] = target
    
    return pd.DataFrame(data)

@pytest.fixture
def temp_data_file(sample_data):
    """Create temporary data file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

class TestModelTrainer:
    """Test model training functionality"""
    
    def test_load_data(self, temp_data_file):
        """Test data loading"""
        trainer = ModelTrainer()
        X, y = trainer.load_data(temp_data_file)
        
        assert X.shape[1] == 5  # 5 features
        assert len(y) == 100    # 100 samples
        assert set(y.unique()) == {0, 1}  # Binary target

    def test_prepare_data(self, sample_data):
        """Test data preparation"""
        trainer = ModelTrainer()
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(X, y)
        
        assert X_train.shape[1] == 5
        assert X_test.shape[1] == 5
        assert len(feature_names) == 5
        assert X_train.shape[0] + X_test.shape[0] == 100

    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    def test_train_model_no_tuning(self, mock_log_metric, mock_log_param, mock_start_run):
        """Test model training without hyperparameter tuning"""
        trainer = ModelTrainer()
        X_train = np.random.random((80, 5))
        y_train = np.random.choice([0, 1], 80)
        
        trainer.train_model(X_train, y_train, tune_hyperparameters=False)
        
        assert trainer.model is not None
        assert hasattr(trainer.model, 'predict')

    def test_evaluate_model(self, sample_data):
        """Test model evaluation"""
        trainer = ModelTrainer()
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        X_train, X_test, y_train, y_test, _ = trainer.prepare_data(X, y)
        trainer.train_model(X_train, y_train, tune_hyperparameters=False)
        
        with patch('mlflow.log_metric'):
            metrics = trainer.evaluate_model(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1

# tests/unit/test_data_validation.py
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.data.validate_data import DataValidator

@pytest.fixture
def good_data():
    """Create good quality data"""
    np.random.seed(42)
    n_samples = 1000
    data = {}
    for i in range(5):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    X = np.column_stack(list(data.values()))
    target = (X[:, 0] + X[:, 1] > 0).astype(int)
    data['target'] = target
    
    return pd.DataFrame(data)

@pytest.fixture
def bad_data():
    """Create poor quality data"""
    n_samples = 100
    data = {}
    
    # Features with missing values
    for i in range(3):
        feature_data = np.random.normal(0, 1, n_samples)
        # Add missing values
        missing_indices = np.random.choice(n_samples, size=30, replace=False)
        feature_data[missing_indices] = np.nan
        data[f'feature_{i}'] = feature_data
    
    # Target with class imbalance
    data['target'] = [1] * 95 + [0] * 5  # 95% class 1, 5% class 0
    
    return pd.DataFrame(data)

class TestDataValidator:
    """Test data validation functionality"""
    
    def test_validate_good_data(self, good_data):
        """Test validation of good quality data"""
        validator = DataValidator()
        results = validator.validate_data_quality(good_data)
        
        assert results['data_quality']['passed'] is True
        assert results['missing_values']['total_missing'] == 0
        assert results['duplicates']['has_duplicates'] is False
        assert results['target_distribution']['is_balanced'] is True

    def test_validate_bad_data(self, bad_data):
        """Test validation of poor quality data"""
        validator = DataValidator()
        results = validator.validate_data_quality(bad_data)
        
        assert results['data_quality']['passed'] is False
        assert results['missing_values']['total_missing'] > 0
        assert results['target_distribution']['is_balanced'] is False

    def test_missing_values_detection(self, bad_data):
        """Test missing values detection"""
        validator = DataValidator()
        missing_info = validator._check_missing_values(bad_data)
        
        assert missing_info['total_missing'] > 0
        assert len(missing_info['columns_with_missing']) > 0

    def test_duplicate_detection(self, good_data):
        """Test duplicate detection"""
        # Add some duplicate rows
        duplicated_data = pd.concat([good_data, good_data.head(10)], ignore_index=True)
        
        validator = DataValidator()
        duplicate_info = validator._check_duplicates(duplicated_data)
        
        assert duplicate_info['has_duplicates'] is True
        assert duplicate_info['duplicate_rows'] == 10

    def test_outlier_detection(self, good_data):
        """Test outlier detection"""
        # Add some extreme outliers
        outlier_data = good_data.copy()
        outlier_data.loc[0, 'feature_0'] = 100  # Extreme value
        
        validator = DataValidator()
        outlier_info = validator._detect_outliers(outlier_data)
        
        assert 'feature_0' in outlier_info
        assert outlier_info['feature_0']['outlier_count'] > 0

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])