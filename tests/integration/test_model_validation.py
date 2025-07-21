import pytest
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class TestModelValidation:
    """Integration tests for model validation"""
    
    @pytest.fixture
    def sample_data(self):
        # Generate sample data for testing
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple decision boundary
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y
        
        return df
    
    def test_model_performance_threshold(self, sample_data):
        """Test that model meets minimum performance threshold"""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Load latest model from MLflow
        model_uri = "models:/test-model/latest"
        try:
            model = mlflow.sklearn.load_model(model_uri)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Assert minimum performance threshold
            assert accuracy > 0.7, f"Model accuracy {accuracy} below threshold 0.7"
            
        except mlflow.exceptions.MlflowException:
            pytest.skip("No model found in registry for testing")
    
    def test_model_feature_compatibility(self, sample_data):
        """Test that model accepts expected features"""
        model_uri = "models:/test-model/latest"
        try:
            model = mlflow.sklearn.load_model(model_uri)
            
            # Test with correct features
            X_sample = sample_data.drop('target', axis=1).iloc[:1]
            prediction = model.predict(X_sample)
            assert len(prediction) == 1
            
            # Test feature count mismatch
            X_wrong = X_sample.iloc[:, :-1]  # Remove one feature
            with pytest.raises(ValueError):
                model.predict(X_wrong)
                
        except mlflow.exceptions.MlflowException:
            pytest.skip("No model found in registry for testing")
    
    def test_model_prediction_range(self, sample_data):
        """Test that predictions are in expected range"""
        model_uri = "models:/test-model/latest"
        try:
            model = mlflow.sklearn.load_model(model_uri)
            X_sample = sample_data.drop('target', axis=1)
            predictions = model.predict(X_sample)
            
            # For classification, predictions should be discrete
            unique_predictions = np.unique(predictions)
            assert len(unique_predictions) <= 10, "Too many unique predictions for classification"
            
            # All predictions should be finite
            assert np.all(np.isfinite(predictions)), "Model produced infinite predictions"
            
        except mlflow.exceptions.MlflowException:
            pytest.skip("No model found in registry for testing")