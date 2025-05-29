import mlflow
import mlflow.pytorch
import torch
import numpy as np
from datetime import datetime, timedelta
import json
import os
import pickle
from pathlib import Path
from behavior_detector import BehaviorClassifier, FeatureExtractor
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import yaml
from dataclasses import dataclass, asdict
import threading
import schedule
import time
from contextlib import contextmanager
import warnings
from torch.utils.data import DataLoader
import wandb
from packaging import version

# Set up logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlops_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    input_size: int = 8
    hidden_size: int = 128
    num_layers: int = 2
    num_classes: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DataDriftMetrics:
    """Metrics for detecting data drift"""
    feature_means: List[float]
    feature_stds: List[float]
    class_distribution: List[float]
    timestamp: str

class ModelRegistry:
    """Model registry for version management"""
    
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict:
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def register_model(self, model_id: str, model_path: str, metrics: Dict, config: Dict):
        """Register a new model version"""
        self.models[model_id] = {
            'path': model_path,
            'metrics': metrics,
            'config': config,
            'registered_at': datetime.now().isoformat(),
            'status': 'staged'
        }
        self._save_registry()
    
    def promote_model(self, model_id: str, stage: str = 'production'):
        """Promote model to production"""
        if model_id in self.models:
            self.models[model_id]['status'] = stage
            self._save_registry()
            logger.info(f"Model {model_id} promoted to {stage}")
    
    def get_production_model(self) -> Optional[str]:
        """Get current production model ID"""
        for model_id, info in self.models.items():
            if info['status'] == 'production':
                return model_id
        return None
    
    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.models, f, indent=2)

class DataValidator:
    """Data quality and drift detection"""
    
    def __init__(self, baseline_stats_path: str = "baseline_stats.pkl"):
        self.baseline_stats_path = baseline_stats_path
        self.baseline_stats = self._load_baseline_stats()
    
    def _load_baseline_stats(self) -> Optional[DataDriftMetrics]:
        if os.path.exists(self.baseline_stats_path):
            with open(self.baseline_stats_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def calculate_data_stats(self, data: np.ndarray, labels: np.ndarray) -> DataDriftMetrics:
        """Calculate baseline statistics for data"""
        return DataDriftMetrics(
            feature_means=data.mean(axis=0).tolist(),
            feature_stds=data.std(axis=0).tolist(),
            class_distribution=np.bincount(labels).tolist(),
            timestamp=datetime.now().isoformat()
        )
    
    def set_baseline(self, data: np.ndarray, labels: np.ndarray):
        """Set baseline statistics"""
        self.baseline_stats = self.calculate_data_stats(data, labels)
        with open(self.baseline_stats_path, 'wb') as f:
            pickle.dump(self.baseline_stats, f)
    
    def detect_drift(self, data: np.ndarray, labels: np.ndarray, threshold: float = 2.0) -> Dict:
        """Detect data drift using statistical tests"""
        if self.baseline_stats is None:
            logger.warning("No baseline stats available for drift detection")
            return {'drift_detected': False, 'reason': 'No baseline'}
        
        current_stats = self.calculate_data_stats(data, labels)
        
        # Check feature drift using z-score
        feature_drift_scores = []
        for i, (baseline_mean, baseline_std, current_mean) in enumerate(
            zip(self.baseline_stats.feature_means, 
                self.baseline_stats.feature_stds, 
                current_stats.feature_means)
        ):
            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / baseline_std
                feature_drift_scores.append(z_score)
            else:
                feature_drift_scores.append(0)
        
        max_drift_score = max(feature_drift_scores)
        drift_detected = max_drift_score > threshold
        
        return {
            'drift_detected': drift_detected,
            'max_drift_score': max_drift_score,
            'feature_drift_scores': feature_drift_scores,
            'threshold': threshold
        }

class EnhancedMLOpsPipeline:
    def __init__(self, 
                 model_dir: str = "models", 
                 data_dir: str = "data",
                 config_path: str = "config.yaml",
                 enable_wandb: bool = False):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.config_path = config_path
        self.metrics_history = []
        self.current_model_version = None
        self.enable_wandb = enable_wandb
        
        # Create necessary directories
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model_registry = ModelRegistry()
        self.data_validator = DataValidator()
        
        # Initialize MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Initialize Weights & Biases if enabled
        if self.enable_wandb:
            try:
                wandb.init(project="behavior-classification", job_type="training")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.enable_wandb = False
        
        # Load configuration
        self.config = self._load_config()
        
        # Background monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
    def _load_config(self) -> ModelConfig:
        """Load configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                return ModelConfig(**config_dict)
        else:
            # Create default config
            default_config = ModelConfig()
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: ModelConfig):
        """Save configuration to YAML file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    @contextmanager
    def error_handler(self, operation: str):
        """Context manager for error handling"""
        try:
            logger.info(f"Starting {operation}")
            yield
            logger.info(f"Completed {operation}")
        except Exception as e:
            logger.error(f"Error in {operation}: {str(e)}")
            raise
    
    def validate_data(self, data: Dict) -> bool:
        """Validate input data quality"""
        try:
            for batch in data['batches']:
                features = batch['features']
                labels = batch['labels']
                
                # Check for NaN values
                if torch.isnan(features).any():
                    logger.error("NaN values found in features")
                    return False
                
                # Check label validity
                if (labels < 0).any() or (labels >= self.config.num_classes).any():
                    logger.error("Invalid label values")
                    return False
                
                # Check feature dimensions
                if features.shape[1] != self.config.input_size:
                    logger.error(f"Feature dimension mismatch: expected {self.config.input_size}, got {features.shape[1]}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
    
    def create_data_hash(self, data: Dict) -> str:
        """Create hash of training data for versioning"""
        data_str = ""
        for batch in data['batches']:
            data_str += str(batch['features'].numpy().tobytes())
            data_str += str(batch['labels'].numpy().tobytes())
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def train_model(self, 
                   train_data: Dict, 
                   validation_data: Dict, 
                   config: Optional[ModelConfig] = None) -> Tuple[BehaviorClassifier, Dict]:
        """Enhanced training with better tracking and validation"""
        
        if config is None:
            config = self.config
        
        with self.error_handler("model training"):
            # Validate data
            if not self.validate_data(train_data) or not self.validate_data(validation_data):
                raise ValueError("Data validation failed")
            
            # Create data hash for versioning
            data_hash = self.create_data_hash(train_data)
            
            # Use nested run for training
            with mlflow.start_run(nested=True):
                # Log parameters and data hash
                mlflow.log_params(config.to_dict())
                mlflow.log_param("data_hash", data_hash)
                
                # Log to wandb if enabled
                if self.enable_wandb:
                    wandb.config.update(config.to_dict())
                
                # Initialize model with better architecture
                model = BehaviorClassifier(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    num_classes=config.num_classes,
                    dropout=config.dropout
                )
                
                # Enhanced optimizer with weight decay
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
                
                # Learning rate scheduler
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=3, factor=0.5
                )
                
                criterion = torch.nn.CrossEntropyLoss()
                
                # Early stopping
                best_val_loss = float('inf')
                patience_counter = 0
                best_metrics = None
                
                for epoch in range(config.epochs):
                    # Training phase
                    model.train()
                    train_loss = 0
                    train_correct = 0
                    train_total = 0
                    
                    for batch in train_data['batches']:
                        optimizer.zero_grad()
                        outputs = model(batch['features'])
                        loss = criterion(outputs, batch['labels'])
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        train_loss += loss.item()
                        
                        # Calculate training accuracy
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += batch['labels'].size(0)
                        train_correct += (predicted == batch['labels']).sum().item()
                    
                    # Validation phase
                    model.eval()
                    val_loss = 0
                    val_predictions = []
                    val_targets = []
                    val_probabilities = []
                    
                    with torch.no_grad():
                        for batch in validation_data['batches']:
                            outputs = model(batch['features'])
                            loss = criterion(outputs, batch['labels'])
                            val_loss += loss.item()
                            
                            probabilities = torch.softmax(outputs, dim=1)
                            val_probabilities.extend(probabilities.cpu().numpy())
                            val_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                            val_targets.extend(batch['labels'].cpu().numpy())
                    
                    # Calculate metrics
                    train_accuracy = 100 * train_correct / train_total
                    val_accuracy = 100 * sum(p == t for p, t in zip(val_predictions, val_targets)) / len(val_targets)
                    
                    # Calculate AUC if binary classification
                    val_auc = None
                    if config.num_classes == 2:
                        val_probabilities_array = np.array(val_probabilities)
                        val_auc = roc_auc_score(val_targets, val_probabilities_array[:, 1])
                    
                    metrics = {
                        'epoch': epoch,
                        'train_loss': train_loss / len(train_data['batches']),
                        'val_loss': val_loss / len(validation_data['batches']),
                        'train_accuracy': train_accuracy,
                        'val_accuracy': val_accuracy,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    }
                    
                    if val_auc is not None:
                        metrics['val_auc'] = val_auc
                    
                    # Log metrics
                    mlflow.log_metrics(metrics)
                    if self.enable_wandb:
                        wandb.log(metrics)
                    
                    logger.info(f"Epoch {epoch}: Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, Val Loss: {metrics['val_loss']:.4f}")
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_metrics = metrics.copy()
                        
                        # Save best model
                        model_path = f"model_epoch_{epoch}"
                        mlflow.pytorch.log_model(model, model_path)
                        self.current_model_version = mlflow.active_run().info.run_id
                    else:
                        patience_counter += 1
                        if patience_counter >= config.early_stopping_patience:
                            logger.info(f"Early stopping triggered at epoch {epoch}")
                            break
                    
                    # Update learning rate
                    scheduler.step(val_loss)
                
                # Log final metrics and model
                final_metrics = {
                    'best_val_loss': best_val_loss,
                    'final_val_accuracy': best_metrics['val_accuracy'],
                    'total_epochs': epoch + 1,
                    'classification_report': classification_report(val_targets, val_predictions, output_dict=True)
                }
                
                mlflow.log_metrics({k: v for k, v in final_metrics.items() if isinstance(v, (int, float))})
                
                # Register model in registry
                self.model_registry.register_model(
                    model_id=self.current_model_version,
                    model_path=f"mlruns/{mlflow.active_run().info.experiment_id}/{self.current_model_version}/artifacts/model",
                    metrics=final_metrics,
                    config=config.to_dict()
                )
                
                return model, final_metrics
    
    def evaluate_model(self, model: BehaviorClassifier, test_data: Dict, model_name: str = "current") -> Dict:
        """Enhanced model evaluation with comprehensive metrics"""
        with self.error_handler("model evaluation"):
            model.eval()
            predictions = []
            targets = []
            probabilities = []
            
            with torch.no_grad():
                for batch in test_data['batches']:
                    outputs = model(batch['features'])
                    probs = torch.softmax(outputs, dim=1)
                    
                    probabilities.extend(probs.cpu().numpy())
                    predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                    targets.extend(batch['labels'].cpu().numpy())
            
            # Comprehensive metrics
            classification_rep = classification_report(targets, predictions, output_dict=True)
            conf_matrix = confusion_matrix(targets, predictions)
            
            metrics = {
                'model_name': model_name,
                'accuracy': classification_rep['accuracy'],
                'precision': classification_rep['weighted avg']['precision'],
                'recall': classification_rep['weighted avg']['recall'],
                'f1_score': classification_rep['weighted avg']['f1-score'],
                'classification_report': classification_rep,
                'confusion_matrix': conf_matrix.tolist(),
                'support': len(targets)
            }
            
            # Add AUC for binary classification
            if len(np.unique(targets)) == 2:
                probabilities_array = np.array(probabilities)
                metrics['auc'] = roc_auc_score(targets, probabilities_array[:, 1])
            
            # Log metrics to history
            self.metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'metrics': metrics
            })
            
            return metrics
    
    def monitor_performance(self, 
                          accuracy_threshold: float = 0.8,
                          drift_threshold: float = 2.0,
                          recent_days: int = 7) -> Dict:
        """Enhanced performance monitoring with drift detection"""
        
        monitoring_results = {
            'performance_degradation': False,
            'data_drift': False,
            'recommendations': []
        }
        
        # Performance monitoring
        if not self.metrics_history:
            monitoring_results['recommendations'].append("No metrics history available")
            return monitoring_results
        
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > datetime.now() - timedelta(days=recent_days)
        ]
        
        if recent_metrics:
            latest_accuracy = recent_metrics[-1]['metrics']['accuracy']
            if latest_accuracy < accuracy_threshold:
                monitoring_results['performance_degradation'] = True
                monitoring_results['recommendations'].append(
                    f"Model accuracy ({latest_accuracy:.2f}) below threshold ({accuracy_threshold})"
                )
        
        return monitoring_results
    
    def auto_retrain(self, 
                    train_data: Dict, 
                    validation_data: Dict,
                    performance_threshold: float = 0.8):
        """Automatic retraining based on performance monitoring"""
        
        monitoring_results = self.monitor_performance(accuracy_threshold=performance_threshold)
        
        if monitoring_results['performance_degradation'] or monitoring_results['data_drift']:
            logger.info("Triggering automatic retraining...")
            
            # Update configuration for retraining
            retrain_config = ModelConfig(
                **self.config.to_dict(),
                epochs=max(self.config.epochs, 15),  # Increase epochs for retraining
                learning_rate=self.config.learning_rate * 0.5  # Lower learning rate
            )
            
            try:
                model, metrics = self.train_model(train_data, validation_data, retrain_config)
                
                # Check if new model is better
                if metrics['final_val_accuracy'] > performance_threshold * 100:
                    self.model_registry.promote_model(self.current_model_version, 'production')
                    logger.info("New model promoted to production")
                else:
                    logger.warning("Retrained model did not meet performance threshold")
                    
            except Exception as e:
                logger.error(f"Auto-retraining failed: {e}")
    
    def start_monitoring(self, 
                        train_data: Dict, 
                        validation_data: Dict,
                        check_interval_hours: int = 24):
        """Start background monitoring"""
        
        def monitoring_job():
            while not self.stop_monitoring.is_set():
                try:
                    self.auto_retrain(train_data, validation_data)
                    time.sleep(check_interval_hours * 3600)  # Convert hours to seconds
                except Exception as e:
                    logger.error(f"Monitoring job error: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        self.monitoring_thread = threading.Thread(target=monitoring_job)
        self.monitoring_thread.start()
        logger.info(f"Started background monitoring (check every {check_interval_hours} hours)")
    
    def stop_background_monitoring(self):
        """Stop background monitoring"""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join()
            logger.info("Stopped background monitoring")
    
    def export_model(self, model: BehaviorClassifier, export_format: str = "torchscript") -> str:
        """Export model to specified format"""
        try:
            if export_format == "torchscript":
                # Get input size from model configuration
                input_size = model.input_size
                
                # Create example input
                example_input = torch.randn(1, input_size)
                
                # Export model
                scripted_model = torch.jit.script(model)
                
                # Save model
                export_path = self.model_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                scripted_model.save(str(export_path))
                
                logger.info(f"Model exported to {export_path}")
                return str(export_path)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise
    
    def save_metrics_history(self, filename: str = "metrics_history.json"):
        """Save enhanced metrics history"""
        history_path = self.data_dir / filename
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Metrics history saved to {history_path}")
    
    def load_metrics_history(self, filename: str = "metrics_history.json"):
        """Load metrics history"""
        history_path = self.data_dir / filename
        try:
            with open(history_path, 'r') as f:
                self.metrics_history = json.load(f)
            logger.info(f"Loaded metrics history from {history_path}")
        except FileNotFoundError:
            logger.warning(f"Metrics history file not found: {history_path}")
            self.metrics_history = []
    
    def generate_report(self) -> str:
        """Generate comprehensive pipeline report"""
        report = {
            'pipeline_status': {
                'current_model_version': self.current_model_version,
                'total_experiments': len(self.metrics_history),
                'last_training': self.metrics_history[-1]['timestamp'] if self.metrics_history else None
            },
            'model_registry': {
                'total_models': len(self.model_registry.models),
                'production_model': self.model_registry.get_production_model()
            },
            'recent_performance': self.metrics_history[-5:] if len(self.metrics_history) >= 5 else self.metrics_history
        }
        
        report_path = self.data_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Pipeline report generated: {report_path}")
        return str(report_path)
    
    def cleanup_old_models(self, keep_last_n: int = 5):
        """Clean up old model artifacts"""
        models_by_date = sorted(
            self.model_registry.models.items(),
            key=lambda x: x[1]['registered_at'],
            reverse=True
        )
        
        for model_id, model_info in models_by_date[keep_last_n:]:
            if model_info['status'] != 'production':
                # Mark for deletion (implement actual deletion logic as needed)
                logger.info(f"Marked model {model_id} for cleanup")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        self.stop_background_monitoring()
        if self.enable_wandb:
            wandb.finish()

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = ModelConfig(
        input_size=8,
        hidden_size=256,
        num_layers=3,
        dropout=0.2,
        learning_rate=0.001,
        epochs=20,
        early_stopping_patience=5
    )
    
    # Initialize enhanced pipeline
    with EnhancedMLOpsPipeline(enable_wandb=False) as pipeline:
        pipeline.save_config(config)
        
        # Example usage (uncomment when you have real data)
        """
        # Load your data
        train_data = load_training_data()
        validation_data = load_validation_data()
        test_data = load_test_data()
        
        # Set baseline for drift detection
        pipeline.data_validator.set_baseline(train_features, train_labels)
        
        # Train model
        model, training_metrics = pipeline.train_model(train_data, validation_data, config)
        
        # Evaluate model
        test_metrics = pipeline.evaluate_model(model, test_data)
        
        # Export model for deployment
        export_path = pipeline.export_model(model, "torchscript")
        
        # Start background monitoring
        pipeline.start_monitoring(train_data, validation_data, check_interval_hours=12)
        
        # Generate report
        report_path = pipeline.generate_report()
        
        # Save metrics
        pipeline.save_metrics_history()
        """
        
        logger.info("Enhanced MLOps Pipeline initialized successfully")