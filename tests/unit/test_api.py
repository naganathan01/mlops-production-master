import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.api.app import app
from src.model.predict import ModelPredictor

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_predictor():
    predictor = Mock(spec=ModelPredictor)
    predictor.predict = AsyncMock(return_value={
        'prediction': 0.85,
        'model_version': 'v1.0.0',
        'confidence': 0.92,
        'timestamp': '2025-07-21T10:00:00'
    })
    predictor.model = Mock()
    predictor.model_version = 'v1.0.0'
    return predictor

class TestHealthEndpoint:
    def test_health_check_healthy(self, client, mock_predictor):
        with patch('src.api.app.predictor', mock_predictor):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'healthy'
            assert data['model_loaded'] is True

    def test_health_check_unhealthy(self, client):
        with patch('src.api.app.predictor', None):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'unhealthy'
            assert data['model_loaded'] is False

class TestPredictionEndpoint:
    def test_prediction_success(self, client, mock_predictor):
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json={
                "features": {"feature1": 1.0, "feature2": 2.0},
                "model_version": "latest"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data['prediction'] == 0.85
            assert data['model_version'] == 'v1.0.0'
            assert data['confidence'] == 0.92

    def test_prediction_empty_features(self, client, mock_predictor):
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json={
                "features": {},
                "model_version": "latest"
            })
            
            assert response.status_code == 400

    def test_prediction_model_error(self, client, mock_predictor):
        mock_predictor.predict.side_effect = Exception("Model error")
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/predict", json={
                "features": {"feature1": 1.0}
            })
            
            assert response.status_code == 500

class TestBatchPrediction:
    def test_batch_prediction_success(self, client, mock_predictor):
        with patch('src.api.app.predictor', mock_predictor):
            response = client.post("/batch-predict", json=[
                {"features": {"feature1": 1.0, "feature2": 2.0}},
                {"features": {"feature1": 2.0, "feature2": 3.0}}
            ])
            
            assert response.status_code == 200
            data = response.json()
            assert data['total'] == 2
            assert data['successful'] == 2

class TestMetricsEndpoint:
    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "ml_requests_total" in response.text

class TestModelInfo:
    def test_model_info_success(self, client, mock_predictor):
        mock_predictor.model_version = 'v1.0.0'
        mock_predictor.model_uri = 'models:/test/1'
        mock_predictor.loaded_at = '2025-07-21T10:00:00'
        mock_predictor.expected_features = ['feature1', 'feature2']
        
        with patch('src.api.app.predictor', mock_predictor):
            response = client.get("/model-info")
            
            assert response.status_code == 200
            data = response.json()
            assert data['model_version'] == 'v1.0.0'
            assert data['features'] == ['feature1', 'feature2']

    def test_model_info_no_model(self, client):
        with patch('src.api.app.predictor', None):
            response = client.get("/model-info")
            assert response.status_code == 503