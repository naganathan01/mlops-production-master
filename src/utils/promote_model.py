# src/utils/promote_model.py
import argparse
import mlflow
from mlflow.tracking import MlflowClient
import structlog

logger = structlog.get_logger()

class ModelPromoter:
    """Handle model promotion between stages"""
    
    def __init__(self, tracking_uri: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def promote_model(self, run_id: str, model_name: str, stage: str, description: str = None) -> bool:
        """Promote model to specified stage"""
        try:
            # Get model version from run
            model_version = self._get_model_version_from_run(run_id, model_name)
            
            if not model_version:
                logger.error("Model version not found for run", run_id=run_id, model_name=model_name)
                return False
            
            # Transition to new stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=stage,
                description=description or f"Promoted from run {run_id}"
            )
            
            logger.info(
                "Model promoted successfully",
                model_name=model_name,
                version=model_version,
                stage=stage,
                run_id=run_id
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to promote model", error=str(e))
            return False
    
    def _get_model_version_from_run(self, run_id: str, model_name: str) -> str:
        """Get model version from run ID"""
        try:
            # Search for model versions with this run_id
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            for version in versions:
                if version.run_id == run_id:
                    return version.version
            
            return None
            
        except Exception as e:
            logger.error("Failed to get model version", error=str(e))
            return None
    
    def archive_old_models(self, model_name: str, stage: str, keep_latest: int = 3):
        """Archive old models in a stage"""
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            
            # Sort by version number (descending)
            versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
            
            # Archive older versions
            for version in versions[keep_latest:]:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived",
                    description=f"Archived old version from {stage}"
                )
                
                logger.info(
                    "Model version archived",
                    model_name=model_name,
                    version=version.version,
                    previous_stage=stage
                )
                
        except Exception as e:
            logger.error("Failed to archive old models", error=str(e))

def main():
    parser = argparse.ArgumentParser(description='Promote ML model')
    parser.add_argument('--run-id', required=True, help='MLflow run ID')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--stage', required=True, choices=['Staging', 'Production'], help='Target stage')
    parser.add_argument('--description', help='Promotion description')
    parser.add_argument('--tracking-uri', help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    promoter = ModelPromoter(args.tracking_uri)
    success = promoter.promote_model(
        args.run_id,
        args.model_name,
        args.stage,
        args.description
    )
    
    if success:
        print(f"✅ Model promoted to {args.stage}")
        
        # Archive old models if promoting to Production
        if args.stage == "Production":
            promoter.archive_old_models(args.model_name, "Production")
        
        exit(0)
    else:
        print(f"❌ Failed to promote model to {args.stage}")
        exit(1)

if __name__ == "__main__":
    main()