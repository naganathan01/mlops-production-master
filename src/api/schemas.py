# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    model_version: Optional[str] = Field("latest", description="Model version to use")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "feature_0": 1.5,
                    "feature_1": 2.3,
                    "feature_2": -0.8
                },
                "model_version": "latest"
            }
        }

class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Model prediction")
    model_version: str = Field(..., description="Model version used")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")
    
class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., description="List of prediction requests")
    
class BatchPredictionResponse(BaseModel):
    results: List[Dict] = Field(..., description="Prediction results")
    total: int = Field(..., description="Total requests processed")
    successful: int = Field(..., description="Successfully processed requests")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="Model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelInfoResponse(BaseModel):
    model_version: str = Field(..., description="Current model version")
    model_uri: str = Field(..., description="Model URI in registry")
    loaded_at: str = Field(..., description="Model load timestamp")
    features: List[str] = Field(..., description="Expected feature names")