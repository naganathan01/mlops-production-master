import os
from typing import Optional
from pydantic import BaseSettings, Field

class Config(BaseSettings):
    """Application configuration with environment variable support"""
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = Field(
        default="http://mlflow:5000",
        description="MLflow tracking server URI"
    )
    MODEL_NAME: str = Field(
        default="production-model",
        description="MLflow registered model name"
    )
    MODEL_STAGE: str = Field(
        default="Production",
        description="MLflow model stage"
    )
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8080)
    API_WORKERS: int = Field(default=4)
    LOG_LEVEL: str = Field(default="INFO")
    
    # Performance Configuration
    MAX_BATCH_SIZE: int = Field(default=100)
    REQUEST_TIMEOUT: int = Field(default=30)
    MODEL_CACHE_SIZE: int = Field(default=3)
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=9090)
    HEALTH_CHECK_INTERVAL: int = Field(default=30)
    
    # Data Configuration
    DATA_PATH: Optional[str] = Field(default=None)
    FEATURE_STORE_URI: Optional[str] = Field(default=None)
    
    # Security Configuration
    API_KEY_HEADER: str = Field(default="X-API-Key")
    ENABLE_CORS: bool = Field(default=True)
    ALLOWED_ORIGINS: list = Field(default=["*"])
    
    # Auto-retraining Configuration
    RETRAIN_THRESHOLD_ACCURACY: float = Field(default=0.95)
    RETRAIN_THRESHOLD_DRIFT: float = Field(default=0.1)
    RETRAIN_SCHEDULE_CRON: str = Field(default="0 2 * * 1")  # Weekly Monday 2 AM
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Global configuration instance
config = Config()