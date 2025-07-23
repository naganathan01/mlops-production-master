# src/model/train.py
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import structlog
from datetime import datetime

logger = structlog.get_logger()

class ModelTrainer:
    """ML Model Training with MLflow tracking"""
    
    def __init__(self, experiment_name: str = "production-model"):
        self.experiment_name = experiment_name
        self.model = None
        self.scaler = StandardScaler()
        
    def setup_experiment(self):
        """Setup MLflow experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow experiment setup complete", name=self.experiment_name)
        except Exception as e:
            logger.error("Failed to setup experiment", error=str(e))
            raise
    
    def load_data(self, data_path: str) -> tuple:
        """Load and prepare training data"""
        try:
            df = pd.read_csv(data_path)
            logger.info("Data loaded successfully", shape=df.shape)
            
            # Separate features and target
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Log data info
            logger.info("Features extracted", n_features=X.shape[1], n_samples=X.shape[0])
            return X, y
            
        except Exception as e:
            logger.error("Failed to load data", error=str(e))
            raise
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Prepare training and validation data"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Data preparation complete", 
                   train_size=X_train_scaled.shape[0], 
                   test_size=X_test_scaled.shape[0])
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist()
    
    def train_model(self, X_train: np.ndarray, y_train: np.Series, tune_hyperparameters: bool = True):
        """Train the model with optional hyperparameter tuning"""
        
        if tune_hyperparameters:
            logger.info("Starting hyperparameter tuning")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Grid search
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info("Hyperparameter tuning complete", 
                       best_params=best_params, 
                       best_cv_score=best_score)
            
            # Log hyperparameters
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_score", best_score)
            
        else:
            # Use default parameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            logger.info("Model training complete with default parameters")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.Series):
        """Evaluate the trained model"""
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report['weighted avg']['precision'])
        mlflow.log_metric("recall", report['weighted avg']['recall'])
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        
        logger.info("Model evaluation complete", 
                   accuracy=accuracy,
                   precision=report['weighted avg']['precision'],
                   recall=report['weighted avg']['recall'])
        
        return {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'feature_importance': feature_importance.tolist()
        }
    
    def save_model(self, model_name: str, feature_names: list):
        """Save model to MLflow registry"""
        try:
            # Log model
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                registered_model_name=model_name,
                input_example=np.random.random((1, len(feature_names))),
                signature=mlflow.models.infer_signature(
                    np.random.random((10, len(feature_names))),
                    self.model.predict(np.random.random((10, len(feature_names))))
                )
            )
            
            # Save scaler separately
            mlflow.log_artifact("scaler.pkl")
            joblib.dump(self.scaler, "scaler.pkl")
            
            # Log additional info
            mlflow.log_param("feature_names", feature_names)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("model_type", "RandomForestClassifier")
            
            logger.info("Model saved successfully", model_name=model_name)
            
        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise
    
    def run_training_pipeline(self, data_path: str, model_name: str, tune_params: bool = True):
        """Complete training pipeline"""
        
        with mlflow.start_run():
            # Log run info
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("training_timestamp", datetime.utcnow().isoformat())
            
            # Load and prepare data
            X, y = self.load_data(data_path)
            X_train, X_test, y_train, y_test, feature_names = self.prepare_data(X, y)
            
            # Train model
            self.train_model(X_train, y_train, tune_params)
            
            # Evaluate model
            metrics = self.evaluate_model(X_test, y_test)
            
            # Save model
            self.save_model(model_name, feature_names)
            
            # Get run info
            run = mlflow.active_run()
            run_id = run.info.run_id
            
            logger.info("Training pipeline complete", 
                       run_id=run_id,
                       accuracy=metrics['accuracy'])
            
            return run_id, metrics

def main():
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--data-path', required=True, help='Path to training data')
    parser.add_argument('--experiment-name', default='production-model', help='MLflow experiment name')
    parser.add_argument('--model-name', default='production-model', help='Model name for registry')
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Setup logging
    from src.utils.logging import configure_logging
    configure_logging()
    
    # Train model
    trainer = ModelTrainer(args.experiment_name)
    trainer.setup_experiment()
    
    run_id, metrics = trainer.run_training_pipeline(
        args.data_path, 
        args.model_name, 
        not args.no_tuning
    )
    
    print(f"✅ Training complete!")
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"MLflow UI: http://localhost:5000")

if __name__ == "__main__":
    main()