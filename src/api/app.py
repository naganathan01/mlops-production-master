# src/api/app.py
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any
import uvicorn
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
import structlog

from src.api.schemas import (
    PredictionRequest, PredictionResponse, 
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse
)
from src.api.middleware import RequestLoggingMiddleware, RateLimitMiddleware
from src.utils.logging import configure_logging
from src.utils.metrics import ModelMetrics, BusinessMetrics

# Configure logging
configure_logging()
logger = structlog.get_logger()

class MLModelService:
    """ML Model serving service"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_version = None
        self.model_uri = None
        self.feature_names = []
        self.loaded_at = None
        self.start_time = time.time()
        
        # Metrics
        self.model_metrics = ModelMetrics()
        self.business_metrics = BusinessMetrics()
        
    async def load_model(self):
        """Load model from MLflow registry"""
        try:
            model_name = os.getenv("MODEL_NAME", "production-model")
            stage = os.getenv("MODEL_STAGE", "Production")
            
            # Try to load from registry
            try:
                model_uri = f"models:/{model_name}/{stage}"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_uri = model_uri
                self.model_version = f"{model_name}:{stage}"
            except Exception:
                # Fallback to latest version
                logger.warning("Failed to load from registry, trying latest version")
                model_uri = f"models:/{model_name}/latest"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_uri = model_uri
                self.model_version = f"{model_name}:latest"
            
            self.loaded_at = datetime.utcnow().isoformat()
            
            # Get feature names (if available)
            self.feature_names = [f"feature_{i}" for i in range(10)]  # Default
            
            # Update metrics
            self.model_metrics.update_model_info(
                self.model_version, self.model_uri, self.loaded_at
            )
            
            logger.info("Model loaded successfully", 
                       model_version=self.model_version,
                       model_uri=self.model_uri)
            
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            # Create a dummy model for testing
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            # Train on dummy data
            X_dummy = np.random.random((100, 10))
            y_dummy = np.random.choice([0, 1], 100)
            self.model.fit(X_dummy, y_dummy)
            self.model_version = "dummy:v1.0"
            self.model_uri = "dummy://model"
            self.loaded_at = datetime.utcnow().isoformat()
            self.feature_names = [f"feature_{i}" for i in range(10)]
            
            logger.warning("Using dummy model for testing")
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction"""
        start_time = time.time()
        
        try:
            # Prepare features
            feature_array = np.array([
                features.get(name, 0.0) for name in self.feature_names
            ]).reshape(1, -1)
            
            # Make prediction
            prediction = float(self.model.predict(feature_array)[0])
            confidence = None
            
            # Get confidence if available
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(feature_array)[0]
                confidence = float(max(prob))
            
            latency = time.time() - start_time
            
            # Record metrics
            self.model_metrics.record_prediction(
                features, prediction, self.model_version, latency, True
            )
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'model_version': self.model_version,
                'timestamp': datetime.utcnow().isoformat(),
                'latency_ms': latency * 1000
            }
            
        except Exception as e:
            latency = time.time() - start_time
            self.model_metrics.record_prediction(
                features, 0.0, self.model_version, latency, False
            )
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return self.model is not None
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        return time.time() - self.start_time

# Initialize service
ml_service = MLModelService()

# Create FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Production ML Model Serving API",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.on_event("startup")
async def startup_event():
    """Initialize the service"""
    logger.info("Starting ML API service")
    await ml_service.load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if ml_service.is_healthy() else "unhealthy",
        model_loaded=ml_service.model is not None,
        uptime=ml_service.get_uptime(),
        version=ml_service.model_version or "unknown"
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get current model information"""
    if not ml_service.is_healthy():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_version=ml_service.model_version,
        model_uri=ml_service.model_uri,
        loaded_at=ml_service.loaded_at,
        features=ml_service.feature_names
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction"""
    if not ml_service.is_healthy():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = ml_service.predict(request.features)
    
    return PredictionResponse(
        prediction=result['prediction'],
        model_version=result['model_version'],
        confidence=result['confidence'],
        timestamp=result['timestamp']
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    if not ml_service.is_healthy():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    successful = 0
    
    for pred_request in request.requests:
        try:
            result = ml_service.predict(pred_request.features)
            results.append(result)
            successful += 1
        except Exception as e:
            results.append({"error": str(e)})
    
    return BatchPredictionResponse(
        results=results,
        total=len(request.requests),
        successful=successful
    )

@app.post("/feedback")
async def record_feedback(
    prediction_id: str,
    actual_outcome: float,
    user_satisfaction: float = None
):
    """Record feedback for a prediction"""
    # This would typically store feedback in a database
    # For now, just log it
    logger.info("Feedback received", 
               prediction_id=prediction_id,
               actual_outcome=actual_outcome,
               user_satisfaction=user_satisfaction)
    
    # Update business metrics if available
    if user_satisfaction is not None:
        ml_service.business_metrics.record_user_satisfaction(0.5, user_satisfaction)
    
    return {"status": "feedback recorded"}

@app.get("/stats")
async def get_stats():
    """Get model statistics"""
    model_stats = ml_service.model_metrics.get_summary_stats()
    business_stats = ml_service.business_metrics.get_business_summary()
    
    return {
        "model_metrics": model_stats,
        "business_metrics": business_stats,
        "service_uptime": ml_service.get_uptime()
    }

@app.post("/reload-model")
async def reload_model():
    """Reload the model (admin endpoint)"""
    try:
        await ml_service.load_model()
        return {"status": "model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    # Start the server
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False
    )