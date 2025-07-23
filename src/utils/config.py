# src/utils/config.py
import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import structlog

logger = structlog.get_logger()

@dataclass
class ModelConfig:
    """Model configuration settings"""
    name: str = "production-model"
    stage: str = "Production"
    version: Optional[str] = None
    uri: Optional[str] = None
    experiment_name: str = "production-model"
    
@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090
    max_batch_size: int = 100
    request_timeout: int = 30
    
@dataclass
class MLflowConfig:
    """MLflow configuration"""
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_root: str = "./mlruns"
    registry_uri: Optional[str] = None
    
@dataclass
class DataConfig:
    """Data configuration"""
    training_data_path: str = "data/sample_training_data.csv"
    test_data_path: str = "data/sample_training_data.csv"
    validation_split: float = 0.2
    random_seed: int = 42
    
@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_prometheus: bool = True
    drift_threshold: float = 2.0
    accuracy_threshold: float = 0.7
    alert_email: Optional[str] = None
    slack_webhook: Optional[str] = None
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    hyperparameter_tuning: bool = True
    cross_validation_folds: int = 5
    max_training_time: int = 3600  # seconds
    early_stopping_patience: int = 10
    model_type: str = "RandomForest"
    
@dataclass
class ApplicationConfig:
    """Complete application configuration"""
    model: ModelConfig
    api: APIConfig
    mlflow: MLflowConfig
    data: DataConfig
    monitoring: MonitoringConfig
    training: TrainingConfig
    
    @classmethod
    def from_env(cls) -> 'ApplicationConfig':
        """Create configuration from environment variables"""
        return cls(
            model=ModelConfig(
                name=os.getenv("MODEL_NAME", "production-model"),
                stage=os.getenv("MODEL_STAGE", "Production"),
                version=os.getenv("MODEL_VERSION"),
                experiment_name=os.getenv("EXPERIMENT_NAME", "production-model")
            ),
            api=APIConfig(
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", "8080")),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
                metrics_port=int(os.getenv("METRICS_PORT", "9090")),
                max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "100")),
                request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30"))
            ),
            mlflow=MLflowConfig(
                tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
                artifact_root=os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns"),
                registry_uri=os.getenv("MLFLOW_REGISTRY_URI")
            ),
            data=DataConfig(
                training_data_path=os.getenv("TRAINING_DATA_PATH", "data/sample_training_data.csv"),
                test_data_path=os.getenv("TEST_DATA_PATH", "data/sample_training_data.csv"),
                validation_split=float(os.getenv("VALIDATION_SPLIT", "0.2")),
                random_seed=int(os.getenv("RANDOM_SEED", "42"))
            ),
            monitoring=MonitoringConfig(
                enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
                drift_threshold=float(os.getenv("DRIFT_THRESHOLD", "2.0")),
                accuracy_threshold=float(os.getenv("ACCURACY_THRESHOLD", "0.7")),
                alert_email=os.getenv("ALERT_EMAIL"),
                slack_webhook=os.getenv("SLACK_WEBHOOK")
            ),
            training=TrainingConfig(
                hyperparameter_tuning=os.getenv("HYPERPARAMETER_TUNING", "true").lower() == "true",
                cross_validation_folds=int(os.getenv("CV_FOLDS", "5")),
                max_training_time=int(os.getenv("MAX_TRAINING_TIME", "3600")),
                early_stopping_patience=int(os.getenv("EARLY_STOPPING_PATIENCE", "10")),
                model_type=os.getenv("MODEL_TYPE", "RandomForest")
            )
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ApplicationConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            return cls(
                model=ModelConfig(**config_dict.get("model", {})),
                api=APIConfig(**config_dict.get("api", {})),
                mlflow=MLflowConfig(**config_dict.get("mlflow", {})),
                data=DataConfig(**config_dict.get("data", {})),
                monitoring=MonitoringConfig(**config_dict.get("monitoring", {})),
                training=TrainingConfig(**config_dict.get("training", {}))
            )
            
        except Exception as e:
            logger.error("Failed to load config from file", error=str(e), path=config_path)
            raise
    
    def to_file(self, config_path: str):
        """Save configuration to JSON file"""
        try:
            config_dict = asdict(self)
            
            # Create directory if it doesn't exist
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info("Configuration saved", path=config_path)
            
        except Exception as e:
            logger.error("Failed to save config", error=str(e), path=config_path)
            raise
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        issues = []
        
        # Validate API config
        if not (1024 <= self.api.port <= 65535):
            issues.append(f"Invalid API port: {self.api.port}")
        
        if not (1024 <= self.api.metrics_port <= 65535):
            issues.append(f"Invalid metrics port: {self.api.metrics_port}")
        
        if self.api.port == self.api.metrics_port:
            issues.append("API port and metrics port cannot be the same")
        
        # Validate data config
        if not (0.0 < self.data.validation_split < 1.0):
            issues.append(f"Invalid validation split: {self.data.validation_split}")
        
        # Validate monitoring config
        if self.monitoring.drift_threshold <= 0:
            issues.append(f"Invalid drift threshold: {self.monitoring.drift_threshold}")
        
        if not (0.0 <= self.monitoring.accuracy_threshold <= 1.0):
            issues.append(f"Invalid accuracy threshold: {self.monitoring.accuracy_threshold}")
        
        # Validate training config
        if self.training.cross_validation_folds < 2:
            issues.append(f"CV folds must be >= 2: {self.training.cross_validation_folds}")
        
        if self.training.max_training_time <= 0:
            issues.append(f"Invalid max training time: {self.training.max_training_time}")
        
        # Check file paths exist
        training_path = Path(self.data.training_data_path)
        if not training_path.exists() and not str(training_path).startswith('data/sample'):
            issues.append(f"Training data file not found: {self.data.training_data_path}")
        
        if issues:
            for issue in issues:
                logger.warning("Configuration validation issue", issue=issue)
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_feature_names(self) -> list:
        """Get expected feature names"""
        # This could be loaded from model metadata or configuration
        return [f"feature_{i}" for i in range(10)]
    
    def get_model_uri(self) -> str:
        """Get the complete model URI"""
        if self.model.uri:
            return self.model.uri
        elif self.model.version:
            return f"models:/{self.model.name}/{self.model.version}"
        else:
            return f"models:/{self.model.name}/{self.model.stage}"

class ConfigManager:
    """Configuration management utilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/app_config.json"
        self._config: Optional[ApplicationConfig] = None
    
    def load_config(self) -> ApplicationConfig:
        """Load configuration with fallback hierarchy"""
        try:
            # Try to load from file first
            if os.path.exists(self.config_path):
                self._config = ApplicationConfig.from_file(self.config_path)
                logger.info("Configuration loaded from file", path=self.config_path)
            else:
                # Fall back to environment variables
                self._config = ApplicationConfig.from_env()
                logger.info("Configuration loaded from environment variables")
            
            # Validate configuration
            if not self._config.validate():
                logger.warning("Configuration validation failed, but continuing")
            
            return self._config
            
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            # Return default configuration as last resort
            self._config = ApplicationConfig.from_env()
            return self._config
    
    def get_config(self) -> ApplicationConfig:
        """Get current configuration (load if not already loaded)"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> ApplicationConfig:
        """Force reload configuration"""
        self._config = None
        return self.load_config()
    
    def save_config(self, config: ApplicationConfig):
        """Save configuration to file"""
        config.to_file(self.config_path)
        self._config = config
    
    def create_default_config(self):
        """Create default configuration file"""
        default_config = ApplicationConfig.from_env()
        
        # Create config directory
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Save default configuration
        default_config.to_file(self.config_path)
        logger.info("Default configuration created", path=self.config_path)
        
        return default_config

# Global configuration manager instance
config_manager = ConfigManager()

def get_config() -> ApplicationConfig:
    """Get application configuration"""
    return config_manager.get_config()

def load_config_from_file(config_path: str) -> ApplicationConfig:
    """Load configuration from specific file"""
    return ApplicationConfig.from_file(config_path)

def load_config_from_env() -> ApplicationConfig:
    """Load configuration from environment variables"""
    return ApplicationConfig.from_env()