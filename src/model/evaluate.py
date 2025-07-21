# src/model/evaluate.py
import argparse
import json
from typing import Dict, Any
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import structlog

logger = structlog.get_logger()

class ModelEvaluator:
    """Model evaluation with comprehensive metrics"""
    
    def __init__(self):
        pass
    
    def load_model(self, model_uri: str):
        """Load model from MLflow"""
        try:
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded for evaluation", model_uri=model_uri)
            return self.model
        except Exception as e:
            logger.error("Failed to load model", error=str(e), model_uri=model_uri)
            raise
    
    def load_test_data(self, test_data_path: str) -> tuple:
        """Load and prepare test data"""
        df = pd.read_csv(test_data_path)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        logger.info("Test data loaded", shape=X.shape)
        return X, y
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Add AUC if binary classification
        unique_classes = len(np.unique(y_test))
        if unique_classes == 2 and hasattr(self.model, 'predict_proba'):
            try:
                y_prob = self.model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            except Exception as e:
                logger.warning("Could not calculate ROC AUC", error=str(e))
        
        logger.info("Model evaluation completed", metrics=metrics)
        return metrics
    
    def compare_with_baseline(self, current_metrics: Dict, baseline_path: str) -> Dict[str, Any]:
        """Compare current model with baseline"""
        try:
            with open(baseline_path, 'r') as f:
                baseline_metrics = json.load(f)
        except FileNotFoundError:
            logger.warning("Baseline metrics file not found", path=baseline_path)
            baseline_metrics = {}
        
        comparison = {
            'current': current_metrics,
            'baseline': baseline_metrics,
            'improvements': {}
        }
        
        # Calculate improvements
        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                improvement = ((current_value - baseline_value) / baseline_value) * 100
                comparison['improvements'][metric] = improvement
        
        return comparison
    
    def generate_report(self, model_uri: str, test_data_path: str, 
                       baseline_path: str = None, output_path: str = None) -> Dict:
        """Generate comprehensive evaluation report"""
        
        # Load model and data
        self.load_model(model_uri)
        X_test, y_test = self.load_test_data(test_data_path)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        # Create report
        report = {
            'model_uri': model_uri,
            'test_samples': len(X_test),
            'metrics': metrics,
            'evaluation_timestamp': pd.Timestamp.utcnow().isoformat()
        }
        
        # Compare with baseline if provided
        if baseline_path:
            comparison = self.compare_with_baseline(metrics, baseline_path)
            report.update(comparison)
        
        # Save report if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info("Evaluation report saved", path=output_path)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Evaluate ML model')
    parser.add_argument('--model-uri', required=True, help='MLflow model URI')
    parser.add_argument('--test-data', required=True, help='Test data path')
    parser.add_argument('--baseline-metrics', help='Baseline metrics JSON file')
    parser.add_argument('--output-report', help='Output report path')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    report = evaluator.generate_report(
        args.model_uri,
        args.test_data,
        args.baseline_metrics,
        args.output_report
    )
    
    print("=== Model Evaluation Report ===")
    print(json.dumps(report, indent=2))
    
    # Check if model passes quality gate
    accuracy = report['metrics']['accuracy']
    if accuracy > 0.7:  # Quality gate threshold
        print(f"✅ Model passes quality gate (accuracy: {accuracy:.3f})")
        exit(0)
    else:
        print(f"❌ Model fails quality gate (accuracy: {accuracy:.3f})")
        exit(1)

if __name__ == "__main__":
    main()