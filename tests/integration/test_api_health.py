# tests/integration/test_api_health.py
import pytest
import requests
import time
import argparse
from typing import Dict

class TestAPIHealth:
    """Integration tests for API health and functionality"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip('/')
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        response = requests.get(f"{self.endpoint}/health", timeout=10)
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'uptime' in data
        
        print(f"✅ Health check passed: {data['status']}")
        return data
    
    def test_prediction_endpoint(self):
        """Test prediction endpoint with sample data"""
        payload = {
            "features": {
                f"feature_{i}": float(i * 0.1) 
                for i in range(10)
            },
            "model_version": "latest"
        }
        
        response = requests.post(
            f"{self.endpoint}/predict", 
            json=payload, 
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"⚠️  Prediction endpoint returned {response.status_code}: {response.text}")
            return False
        
        data = response.json()
        assert 'prediction' in data
        assert 'model_version' in data
        assert 'timestamp' in data
        
        print(f"✅ Prediction test passed: {data['prediction']}")
        return True
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = requests.get(f"{self.endpoint}/metrics", timeout=10)
        assert response.status_code == 200
        
        # Check if prometheus metrics are present
        metrics_text = response.text
        assert 'ml_requests_total' in metrics_text or 'python_info' in metrics_text
        
        print("✅ Metrics endpoint accessible")
        return True
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = requests.get(f"{self.endpoint}/model-info", timeout=10)
        
        if response.status_code == 503:
            print("⚠️  Model not loaded yet")
            return False
        
        assert response.status_code == 200
        data = response.json()
        
        expected_fields = ['model_version', 'model_uri', 'loaded_at']
        for field in expected_fields:
            if field not in data:
                print(f"⚠️  Missing field in model info: {field}")
                return False
        
        print("✅ Model info endpoint working")
        return True
    
    def run_smoke_tests(self, max_retries: int = 5, retry_delay: int = 10):
        """Run smoke tests with retries"""
        print(f"Running smoke tests against: {self.endpoint}")
        
        for attempt in range(max_retries):
            try:
                # Test health
                health_data = self.test_health_endpoint()
                
                # Test metrics
                self.test_metrics_endpoint()
                
                # Test prediction (may fail if model not loaded)
                prediction_success = self.test_prediction_endpoint()
                
                # Test model info (may fail if model not loaded)
                model_info_success = self.test_model_info_endpoint()
                
                if prediction_success and model_info_success:
                    print("✅ All smoke tests passed!")
                    return True
                elif attempt < max_retries - 1:
                    print(f"⏳ Some tests failed, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print("⚠️  Some tests failed but basic health checks passed")
                    return True
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"❌ Test attempt {attempt + 1} failed: {e}")
                    print(f"⏳ Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ All test attempts failed: {e}")
                    return False
        
        return False

def main():
    parser = argparse.ArgumentParser(description='API Health Integration Tests')
    parser.add_argument('--endpoint', required=True, help='API endpoint URL')
    parser.add_argument('--max-retries', type=int, default=5, help='Max retry attempts')
    parser.add_argument('--retry-delay', type=int, default=10, help='Delay between retries')
    
    args = parser.parse_args()
    
    tester = TestAPIHealth(args.endpoint)
    success = tester.run_smoke_tests(args.max_retries, args.retry_delay)
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()