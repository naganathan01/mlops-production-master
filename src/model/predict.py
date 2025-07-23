# src/model/predict.py
import argparse
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from typing import Dict, List, Any
import structlog

logger = structlog.get_logger()

class ModelPredictor:
    """ML Model prediction service"""
    
    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.model = None
        self.feature_names = []
        
    def load_model(self):
        """Load model from MLflow"""
        try:
            self.model = mlflow.sklearn.load_model(self.model_uri)
            logger.info("Model loaded successfully", model_uri=self.model_uri)
            
            # Try to get feature names from model signature
            try:
                import mlflow.models
                model_info = mlflow.models.get_model_info(self.model_uri)
                if model_info.signature and model_info.signature.inputs:
                    self.feature_names = [input.name for input in model_info.signature.inputs.inputs]
                else:
                    # Default feature names
                    self.feature_names = [f"feature_{i}" for i in range(10)]
            except:
                self.feature_names = [f"feature_{i}" for i in range(10)]
            
            logger.info("Features loaded", feature_names=self.feature_names)
            
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make single prediction"""
        try:
            # Prepare feature array in correct order
            feature_array = np.array([
                features.get(name, 0.0) for name in self.feature_names
            ]).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(feature_array)[0]
            
            # Get prediction probability if available
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(feature_array)[0]
                confidence = float(max(proba))
            
            return {
                'prediction': float(prediction),
                'confidence': confidence,
                'features_used': list(features.keys()),
                'model_uri': self.model_uri
            }
            
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise
    
    def predict_batch(self, feature_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        try:
            # Prepare feature matrix
            feature_matrix = np.array([
                [features.get(name, 0.0) for name in self.feature_names]
                for features in feature_list
            ])
            
            # Make predictions
            predictions = self.model.predict(feature_matrix)
            
            # Get probabilities if available
            confidences = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_matrix)
                confidences = [float(max(prob)) for prob in probabilities]
            
            # Format results
            results = []
            for i, prediction in enumerate(predictions):
                result = {
                    'prediction': float(prediction),
                    'confidence': confidences[i] if confidences else None,
                    'features_used': list(feature_list[i].keys()),
                    'index': i
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error("Batch prediction failed", error=str(e))
            raise
    
    def predict_from_file(self, input_file: str, output_file: str = None) -> List[Dict[str, Any]]:
        """Make predictions from input file"""
        try:
            # Read input file
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith('.json'):
                with open(input_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            logger.info("Input data loaded", shape=df.shape)
            
            # Convert to feature dictionaries
            feature_list = df.to_dict('records')
            
            # Make predictions
            results = self.predict_batch(feature_list)
            
            # Add input features to results
            for i, result in enumerate(results):
                result['input_features'] = feature_list[i]
            
            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info("Results saved", output_file=output_file)
            
            return results
            
        except Exception as e:
            logger.error("File prediction failed", error=str(e))
            raise

def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model-uri', required=True, help='MLflow model URI')
    parser.add_argument('--input-path', help='Input data file (CSV or JSON)')
    parser.add_argument('--output-path', help='Output file for predictions')
    parser.add_argument('--features', help='JSON string of features for single prediction')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for large files')
    
    args = parser.parse_args()
    
    # Setup logging
    from src.utils.logging import configure_logging
    configure_logging()
    
    # Initialize predictor
    predictor = ModelPredictor(args.model_uri)
    predictor.load_model()
    
    if args.features:
        # Single prediction from command line
        try:
            features = json.loads(args.features)
            result = predictor.predict_single(features)
            print("Prediction Result:")
            print(json.dumps(result, indent=2))
            
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for features")
            exit(1)
            
    elif args.input_path:
        # Batch prediction from file
        try:
            results = predictor.predict_from_file(args.input_path, args.output_path)
            
            print(f"Predictions completed for {len(results)} samples")
            if args.output_path:
                print(f"Results saved to: {args.output_path}")
            else:
                print("Sample results:")
                for i, result in enumerate(results[:3]):  # Show first 3 results
                    print(f"Sample {i+1}: {result['prediction']}")
                    
        except Exception as e:
            print(f"Error processing file: {e}")
            exit(1)
            
    else:
        # Interactive mode
        print("Interactive prediction mode")
        print(f"Expected features: {predictor.feature_names}")
        print("Enter feature values (or 'quit' to exit):")
        
        while True:
            try:
                user_input = input("\nEnter JSON features: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                features = json.loads(user_input)
                result = predictor.predict_single(features)
                
                print("Result:")
                print(f"  Prediction: {result['prediction']}")
                if result['confidence']:
                    print(f"  Confidence: {result['confidence']:.3f}")
                
            except json.JSONDecodeError:
                print("Invalid JSON format. Please try again.")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()