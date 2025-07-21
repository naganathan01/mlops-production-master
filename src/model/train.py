import argparse
import os
from datetime import datetime
from typing import Dict, Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import structlog

logger = structlog.get_logger()

class ModelTrainer:
    """Production model training with MLflow tracking"""
    
    def __init__(self, experiment_name: str, model_name: str):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.scaler = StandardScaler()
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
    def load_and_preprocess_data(self, data_path: str) -> tuple:
        """Load and preprocess training data"""
        logger.info("Loading training data", path=data_path)
        
        # Load data (assuming CSV for this example)
        df = pd.read_csv(data_path)
        
        # Feature engineering
        X = self._feature_engineering(df)
        y = df['target']  # Assuming target column exists
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(
            "Data preprocessing complete",
            train_shape=X_train_scaled.shape,
            test_shape=X_test_scaled.shape
        )
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        # Drop target column and any ID columns
        feature_columns = [col for col in df.columns if col not in ['target', 'id']]
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Add engineered features
        if 'feature1' in X.columns and 'feature2' in X.columns:
            X['feature1_feature2_ratio'] = X['feature1'] / (X['feature2'] + 1e-6)
        
        logger.info("Feature engineering complete", features=list(X.columns))
        return X
    
    def train_model(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train model with hyperparameter tuning"""
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("data_version", datetime.now().strftime("%Y-%m-%d"))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42)
            
            logger.info("Starting hyperparameter tuning")
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            
            # Log best parameters
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Log metrics
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            # Log model
            mlflow.sklearn.log_model(
                best_model, 
                "model",
                registered_model_name=self.model_name,
                metadata={
                    "features": list(range(X_train.shape[1])),  # Feature names
                    "scaler": "StandardScaler"
                }
            )
            
            # Log scaler
            mlflow.sklearn.log_model(self.scaler, "scaler")
            
            # Log feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance_dict = {
                    f"feature_{i}": imp 
                    for i, imp in enumerate(best_model.feature_importances_)
                }
                mlflow.log_metrics(importance_dict)
            
            logger.info(
                "Model training complete",
                run_id=run.info.run_id,
                metrics=metrics
            )
            
            # Set environment variable for pipeline
            os.environ['MLFLOW_RUN_ID'] = run.info.run_id
            
            return {
                'run_id': run.info.run_id,
                'model': best_model,
                'metrics': metrics
            }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--model-name', required=True)
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.experiment_name, args.model_name)
    X_train, X_test, y_train, y_test = trainer.load_and_preprocess_data(args.data_path)
    result = trainer.train_model(X_train, y_train, X_test, y_test)
    
    print(f"Training complete. Run ID: {result['run_id']}")
    print(f"Metrics: {result['metrics']}")

if __name__ == "__main__":
    main()