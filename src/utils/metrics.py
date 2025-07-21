# src/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Dict, Any
import time
import numpy as np
from collections import deque, defaultdict
import structlog

logger = structlog.get_logger()

class ModelMetrics:
    """Custom metrics for ML model monitoring"""
    
    def __init__(self):
        # Prometheus metrics
        self.prediction_counter = Counter(
            'ml_predictions_total', 
            'Total predictions made', 
            ['model_version', 'status']
        )
        
        self.prediction_latency = Histogram(
            'ml_prediction_latency_seconds',
            'Prediction latency in seconds'
        )
        
        self.prediction_values = Histogram(
            'ml_prediction_values',
            'Distribution of prediction values',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Current model accuracy'
        )
        
        self.drift_score = Gauge(
            'ml_drift_score',
            'Model drift detection score'
        )
        
        self.model_info = Info(
            'ml_model_info',
            'Information about the current model'
        )
        
        # Internal tracking
        self.prediction_history = deque(maxlen=1000)
        self.feature_history = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_stats = {}
    
    def record_prediction(self, features: Dict[str, float], 
                         prediction: float, model_version: str, 
                         latency: float, success: bool = True):
        """Record a prediction with all relevant metrics"""
        
        # Update prometheus metrics
        status = 'success' if success else 'error'
        self.prediction_counter.labels(
            model_version=model_version, 
            status=status
        ).inc()
        
        if success:
            self.prediction_latency.observe(latency)
            self.prediction_values.observe(prediction)
            
            # Store for drift detection
            self.prediction_history.append(prediction)
            for feature_name, value in features.items():
                self.feature_history[feature_name].append(value)
    
    def update_model_info(self, model_version: str, model_uri: str, 
                         loaded_at: str):
        """Update model information metrics"""
        self.model_info.info({
            'version': model_version,
            'uri': model_uri,
            'loaded_at': loaded_at
        })
    
    def calculate_drift_score(self) -> float:
        """Calculate and update drift score"""
        if len(self.prediction_history) < 100 or not self.baseline_stats:
            return 0.0
        
        current_mean = np.mean(list(self.prediction_history))
        baseline_mean = self.baseline_stats.get('prediction_mean', current_mean)
        baseline_std = self.baseline_stats.get('prediction_std', 1.0)
        
        # Normalized drift score
        if baseline_std > 0:
            drift_score = abs(current_mean - baseline_mean) / baseline_std
        else:
            drift_score = 0.0
        
        # Cap at reasonable maximum
        drift_score = min(drift_score, 5.0)
        
        self.drift_score.set(drift_score)
        return drift_score
    
    def set_baseline_stats(self, prediction_mean: float, prediction_std: float,
                          feature_stats: Dict[str, Dict[str, float]]):
        """Set baseline statistics for drift detection"""
        self.baseline_stats = {
            'prediction_mean': prediction_mean,
            'prediction_std': prediction_std,
            'feature_stats': feature_stats
        }
        
        logger.info("Baseline statistics updated for drift detection")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for monitoring dashboard"""
        if not self.prediction_history:
            return {}
        
        predictions = list(self.prediction_history)
        return {
            'total_predictions': len(predictions),
            'avg_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions),
            'drift_score': self.calculate_drift_score()
        }

class BusinessMetrics:
    """Track business-specific metrics"""
    
    def __init__(self):
        self.conversion_rate = Gauge(
            'business_conversion_rate',
            'Business conversion rate from ML predictions'
        )
        
        self.revenue_impact = Counter(
            'business_revenue_impact_total',
            'Total revenue impact from ML predictions'
        )
        
        self.user_satisfaction = Histogram(
            'business_user_satisfaction_score',
            'User satisfaction with predictions',
            buckets=[1.0, 2.0, 3.0, 4.0, 5.0]
        )
        
        self.prediction_accuracy_realtime = Gauge(
            'business_prediction_accuracy_realtime',
            'Real-time prediction accuracy based on feedback'
        )
        
        # Internal tracking
        self.conversions = deque(maxlen=1000)
        self.revenue_events = deque(maxlen=1000)
        self.satisfaction_scores = deque(maxlen=1000)
        self.accuracy_feedback = deque(maxlen=1000)
    
    def record_conversion(self, prediction_confidence: float, converted: bool):
        """Record a conversion event"""
        self.conversions.append({
            'confidence': prediction_confidence,
            'converted': converted,
            'timestamp': time.time()
        })
        
        # Update conversion rate (last 100 high-confidence predictions)
        recent_high_confidence = [
            c for c in list(self.conversions)[-100:]
            if c['confidence'] > 0.7
        ]
        
        if recent_high_confidence:
            conversion_rate = sum(c['converted'] for c in recent_high_confidence) / len(recent_high_confidence)
            self.conversion_rate.set(conversion_rate)
    
    def record_revenue_impact(self, prediction_confidence: float, revenue: float):
        """Record revenue impact from a prediction"""
        if prediction_confidence > 0.6:  # Only count medium+ confidence predictions
            self.revenue_impact.inc(revenue)
            self.revenue_events.append({
                'confidence': prediction_confidence,
                'revenue': revenue,
                'timestamp': time.time()
            })
    
    def record_user_satisfaction(self, prediction_confidence: float, 
                               satisfaction_score: float):
        """Record user satisfaction with a prediction"""
        self.user_satisfaction.observe(satisfaction_score)
        self.satisfaction_scores.append({
            'confidence': prediction_confidence,
            'satisfaction': satisfaction_score,
            'timestamp': time.time()
        })
    
    def record_prediction_feedback(self, prediction: float, actual_outcome: float):
        """Record feedback on prediction accuracy"""
        accuracy = 1.0 - abs(prediction - actual_outcome)  # Simple accuracy measure
        self.accuracy_feedback.append({
            'prediction': prediction,
            'actual': actual_outcome,
            'accuracy': accuracy,
            'timestamp': time.time()
        })
        
        # Update real-time accuracy (last 50 feedback events)
        if len(self.accuracy_feedback) >= 10:
            recent_accuracy = np.mean([
                f['accuracy'] for f in list(self.accuracy_feedback)[-50:]
            ])
            self.prediction_accuracy_realtime.set(recent_accuracy)
    
    def get_business_summary(self) -> Dict[str, Any]:
        """Get business metrics summary"""
        recent_conversions = [c for c in self.conversions if c['confidence'] > 0.7]
        recent_revenue = sum(r['revenue'] for r in list(self.revenue_events)[-100:])
        recent_satisfaction = [s['satisfaction'] for s in list(self.satisfaction_scores)[-100:]]
        recent_accuracy = [f['accuracy'] for f in list(self.accuracy_feedback)[-50:]]
        
        return {
            'conversion_rate': len([c for c in recent_conversions if c['converted']]) / max(len(recent_conversions), 1),
            'total_revenue_impact': recent_revenue,
            'avg_satisfaction': np.mean(recent_satisfaction) if recent_satisfaction else 0,
            'prediction_accuracy': np.mean(recent_accuracy) if recent_accuracy else 0,
            'total_feedback_events': len(self.accuracy_feedback)
        }