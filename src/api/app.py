import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import mlflow
import pandas as pd
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from src.utils.config import Config
from src.utils.metrics import ModelMetrics
from src.model.predict import ModelPredictor

# Configure structured logging
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total ML requests', ['model_version', 'status'])
REQUEST_DURATION = Histogram('ml_request_duration_seconds', 'Request duration')
PREDICTION_VALUES = Histogram('ml_prediction_values', 'Distribution of prediction values')

class PredictionRequest(BaseModel):
    features: Dict[str, float]
    model_version: Optional[str] = "latest"

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    confidence: Optional[float] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: float
    version: str

# Global variables
predictor: ModelPredictor = None
start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global predictor
    
    # Startup
    logger.info("Starting ML service...")
    try:
        predictor = ModelPredictor()
        await predictor.load_model()
        logger.info("Model loaded successfully", model_version=predictor.model_version)
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML service...")

app = FastAPI(
    title="Production ML API",
    description="Production-ready ML model serving with monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests with structured logging"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        "Request processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    return response

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Make predictions with comprehensive monitoring"""
    start_time = time.time()
    
    try:
        # Validate input
        if not request.features:
            raise HTTPException(status_code=400, detail="Features cannot be empty")
        
        # Make prediction
        result = await predictor.predict(request.features, request.model_version)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_COUNT.labels(
            model_version=result['model_version'], 
            status='success'
        ).inc()
        REQUEST_DURATION.observe(duration)
        PREDICTION_VALUES.observe(result['prediction'])
        
        # Log prediction (async to avoid blocking)
        background_tasks.add_task(
            log_prediction,
            request.features,
            result,
            duration
        )
        
        return PredictionResponse(
            prediction=result['prediction'],
            model_version=result['model_version'],
            confidence=result.get('confidence'),
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(
            model_version=request.model_version, 
            status='error'
        ).inc()
        logger.error("Prediction failed", error=str(e), features=request.features)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction endpoint for efficiency"""
    results = []
    
    for req in requests:
        try:
            result = await predictor.predict(req.features, req.model_version)
            results.append(result)
        except Exception as e:
            logger.error("Batch prediction item failed", error=str(e))
            results.append({"error": str(e)})
    
    return {"results": results, "total": len(requests), "successful": len([r for r in results if "error" not in r])}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    uptime = time.time() - start_time
    model_loaded = predictor is not None and predictor.model is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        uptime=uptime,
        version=predictor.model_version if predictor else "unknown"
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model-info")
async def model_info():
    """Get current model information"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": predictor.model_version,
        "model_uri": predictor.model_uri,
        "loaded_at": predictor.loaded_at,
        "features": predictor.expected_features
    }

async def log_prediction(features: Dict, result: Dict, duration: float):
    """Async logging function"""
    logger.info(
        "Prediction made",
        features=features,
        prediction=result['prediction'],
        model_version=result['model_version'],
        duration=duration,
        confidence=result.get('confidence')
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)