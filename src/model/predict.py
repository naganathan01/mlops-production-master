import asyncio
import pickle
import time
from datetime import datetime
from typing import Dict, Any, Optional

import mlflow
import pandas as pd
import structlog
from sklearn.base import BaseEstimator

from src.utils.config import Config

logger = structlog.get_logger()

class ModelPredictor:
    """Thread-safe model predictor with MLflow integration"""
    
    def __init__(self):
        self.model: Optional[BaseEstimator] = None
        self.model_version: Optional[str] = None
        self.model_uri: Optional[str] = None
        self.loaded_at: Optional[str] = None
        self.expected_features: Optional[list] = None
        self.config = Config()
        
    async def load_model(self, version: str = "latest"):
        """Load model from MLflow registry"""
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
            
            # Load model
            model_name = self.config.MODEL_NAME
            if version == "latest":
                model_uri = f"models:/{model_name}/Production"
            else:
                model_uri = f"models:/{model_name}/{version}"
            
            logger.info("Loading model", model_uri=model_uri)
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                mlflow.sklearn.load_model, 
                model_uri
            )
            
            self.model_uri = model_uri
            self.model_version = version
            self.loaded_at = datetime.utcnow().isoformat()
            
            # Get expected features (assuming they're stored as model metadata)
            model_info = mlflow.models.get_model_info(model_uri)
            self.expected_features = getattr(
                model_info.flavors.get('sklearn', {}), 
                'serialization_format', 
                []
            )
            
            logger.info(
                "Model loaded successfully",
                model_version=self.model_version,
                model_uri=self.model_uri
            )
            
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise
    
    async def predict(
        self, 
        features: Dict[str, float], 
        requested_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make prediction with the loaded model"""
        
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Reload model if different version requested
        if requested_version and requested_version != self.model_version:
            await self.load_model(requested_version)
        
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # Validate features
            self._validate_features(df)
            
            # Make prediction in thread pool
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                None,
                self.model.predict,
                df
            )
            
            # Calculate confidence if model supports predict_proba
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                proba = await loop.run_in_executor(
                    None,
                    self.model.predict_proba,
                    df
                )
                confidence = float(max(proba[0]))
            
            return {
                'prediction': float(prediction[0]),
                'model_version': self.model_version,
                'confidence': confidence,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Prediction failed", error=str(e), features=features)
            raise
    
    def _validate_features(self, df: pd.DataFrame):
        """Validate input features"""
        if self.expected_features:
            missing_features = set(self.expected_features) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            extra_features = set(df.columns) - set(self.expected_features)
            if extra_features:
                logger.warning("Extra features provided", extra_features=list(extra_features))