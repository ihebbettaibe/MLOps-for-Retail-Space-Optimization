import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, f1_score
import wandb
from tqdm import tqdm
from datetime import datetime
import os
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MLflow
mlflow.set_tracking_uri("file:./mlruns")

class FeatureExtractor:
    """Feature extractor for behavior analysis"""
    
    def extract_features(self, positions: List, timestamps: List) -> np.ndarray:
        """Extract features from position and timestamp data"""
        if len(positions) < 2:
            # Return default features for single point
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        positions = np.array(positions)
        timestamps = np.array(timestamps)
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                v = np.sqrt(dx*dx + dy*dy) / dt
                velocities.append(v)
        
        if not velocities:
            velocities = [0.0]
        
        velocities = np.array(velocities)
        
        # Calculate accelerations
        accelerations = []
        for i in range(1, len(velocities)):
            dt = timestamps[i+1] - timestamps[i]
            if dt > 0:
                dv = velocities[i] - velocities[i-1]
                a = dv / dt
                accelerations.append(a)
        
        if not accelerations:
            accelerations = [0.0]
        
        accelerations = np.array(accelerations)
        
        # Extract statistical features
        features = [
            np.mean(velocities),     # mean velocity
            np.std(velocities),      # velocity std
            np.max(velocities),      # max velocity
            np.mean(accelerations),  # mean acceleration
            np.std(accelerations),   # acceleration std
            len(positions),          # path length
            np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))),  # total distance
            timestamps[-1] - timestamps[0]  # total time
        ]
        
        return np.array(features, dtype=np.float32)

class BehaviorClassifier(nn.Module):
    """Very simple neural network for behavior classification"""
    
    def __init__(self, input_size: int, num_classes: int = 4):
        super(BehaviorClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, num_classes)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.float()
        return self.classifier(x)

class ModelConfig:
    """Configuration class for model parameters"""
    
    def __init__(self, **kwargs):
        self.input_size = kwargs.get('input_size', 8)
        self.hidden_size = kwargs.get('hidden_size', 32)
        self.num_layers = kwargs.get('num_layers', 1)
        self.num_classes = kwargs.get('num_classes', 4)
        self.dropout = kwargs.get('dropout', 0.1)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 16)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 15)
    
    def to_dict(self):
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience
        }

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ModelConfig,
    class_weights: torch.Tensor,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[nn.Module, Dict]:
    """
    Train the behavior classification model with simple training process
    """
    model = model.to(device)
    class_weights = class_weights.float()
    
    # Use weighted loss for imbalanced classes
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Use simple SGD optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    logger.info(f"Training on device: {device}")
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs} [Train]')
        for features, labels in train_pbar:
            features = features.float().to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': train_loss / (train_pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{config.epochs} [Val]')
            for features, labels in val_pbar:
                features = features.float().to(device)
                labels = labels.long().to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': val_loss / (val_pbar.n + 1),
                    'acc': 100. * val_correct / val_total
                })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_acc = 100. * val_correct / val_total
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "train_accuracy": epoch_train_acc,
            "val_accuracy": epoch_val_acc
        }, step=epoch)
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        # Early stopping based on validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model with validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model state
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    
    # Calculate final metrics
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.float().to(device)
            labels = labels.long().to(device)
            
            outputs = model(features)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate classification metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=0)
    
    classification_rep = classification_report(all_targets, all_predictions, output_dict=True, zero_division=0)
    
    metrics = {
        'final_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'best_val_acc': best_val_acc,
        'classification_report': classification_rep,
        'history': history
    }
    
    return model, metrics

def balance_dataset(X, y, min_samples_per_class=50):
    """
    Balance the dataset by augmenting underrepresented classes with more aggressive augmentation
    """
    logger.info("Balancing dataset...")
    
    # Count samples per class
    class_counts = Counter(y)
    logger.info(f"Original class distribution: {dict(class_counts)}")
    
    # Group samples by class
    class_samples = {}
    for i, label in enumerate(y):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(i)
    
    # Augment underrepresented classes
    new_X = list(X)
    new_y = list(y)
    
    for class_label, indices in class_samples.items():
        if len(indices) < min_samples_per_class:
            needed = min_samples_per_class - len(indices)
            logger.info(f"Class {class_label}: augmenting {needed} samples")
            
            # Multiple augmentation techniques
            for _ in range(needed):
                # Pick a random sample from this class
                idx = np.random.choice(indices)
                original_sample = X[idx].copy()
                
                # Random augmentation
                augmented_sample = original_sample.copy()
                
                # Add noise (10% of std dev)
                noise_scale = 0.1 * np.std(X, axis=0)
                noise = np.random.normal(0, noise_scale, original_sample.shape)
                augmented_sample += noise
                
                # Random scaling (0.8 to 1.2)
                scale = np.random.uniform(0.8, 1.2)
                augmented_sample *= scale
                
                # Random rotation (for position features)
                if len(augmented_sample) >= 2:
                    angle = np.random.uniform(-0.2, 0.2)
                    cos_angle = np.cos(angle)
                    sin_angle = np.sin(angle)
                    augmented_sample[0] = augmented_sample[0] * cos_angle - augmented_sample[1] * sin_angle
                    augmented_sample[1] = augmented_sample[0] * sin_angle + augmented_sample[1] * cos_angle
                
                new_X.append(augmented_sample)
                new_y.append(class_label)
    
    X = np.array(new_X)
    y = np.array(new_y)
    
    # Log new distribution
    new_class_counts = Counter(y)
    logger.info(f"Balanced class distribution: {dict(new_class_counts)}")
    
    return X, y

def load_training_data(data_path: str = "data/training_data.json"):
    """Load and preprocess training data"""
    logger.info(f"Loading training data from {data_path}")
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data")
    except FileNotFoundError:
        logger.error(f"Training data file not found: {data_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {data_path}")
        raise
    
    if not data.get('tracks'):
        logger.error("No tracks found in training data")
        raise ValueError("Empty training data")
    
    logger.info(f"Found {len(data['tracks'])} tracks in the data")
    
    # Get video metadata
    fps = data.get('metadata', {}).get('fps', 30.0)
    frame_time = 1.0 / fps
    
    # Convert data to features and labels
    features = []
    labels = []
    
    feature_extractor = FeatureExtractor()
    
    # Track statistics for debugging
    total_tracks = len(data['tracks'])
    skipped_tracks = 0
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 4 classes
    
    for i, track in enumerate(data['tracks']):
        try:
            # Validate track data
            if not track.get('positions'):
                logger.warning(f"Track {i} has no positions")
                skipped_tracks += 1
                continue
            
            # Generate timestamps based on frame rate
            positions = track['positions']
            timestamps = [j * frame_time for j in range(len(positions))]
            
            # Convert positions to the correct format
            positions_list = []
            for pos in positions:
                if isinstance(pos, dict):
                    if 'x' not in pos or 'y' not in pos:
                        logger.warning(f"Invalid position format in track {i}")
                        continue
                    positions_list.append([float(pos['x']), float(pos['y'])])
                else:
                    if len(pos) < 2:
                        logger.warning(f"Invalid position format in track {i}")
                        continue
                    positions_list.append([float(pos[0]), float(pos[1])])
            
            if not positions_list:
                logger.warning(f"No valid positions in track {i}")
                skipped_tracks += 1
                continue
            
            # Get label from track data, or calculate based on movement
            label = None
            if 'behavior' in track and track['behavior'] is not None:
                # Use provided label
                label = int(track['behavior'])
                logger.info(f"Track {i} using provided behavior label: {label}")
            elif 'label' in track and track['label'] is not None:
                label = int(track['label'])
                logger.info(f"Track {i} using provided label: {label}")
            else:
                # Calculate label based on movement speed
                if len(positions_list) > 1:
                    # Calculate total distance moved
                    total_distance = 0
                    for j in range(1, len(positions_list)):
                        dx = positions_list[j][0] - positions_list[j-1][0]
                        dy = positions_list[j][1] - positions_list[j-1][1]
                        total_distance += np.sqrt(dx*dx + dy*dy)
                    
                    # Calculate average speed
                    total_time = timestamps[-1] - timestamps[0]
                    avg_speed = total_distance / total_time if total_time > 0 else 0
                    
                    # Simple speed-based classification
                    if avg_speed < 10:  # Very slow
                        label = 0  # Standing
                    elif avg_speed < 30:  # Slow
                        label = 1  # Walking
                    elif avg_speed < 60:  # Medium
                        label = 2  # Running
                    else:  # Fast
                        label = 3  # Sprinting
                    
                    logger.info(f"Track {i} calculated label based on speed {avg_speed:.2f}: {label}")
                else:
                    label = 0  # Default label for short tracks
                    logger.info(f"Track {i} using default label for short track: {label}")
            
            # Skip if label is still None
            if label is None:
                logger.warning(f"Could not determine label for track {i}")
                skipped_tracks += 1
                continue
            
            # Extract features
            track_features = feature_extractor.extract_features(
                positions=positions_list,
                timestamps=timestamps
            )
            
            if track_features is None:
                logger.warning(f"Feature extraction failed for track {i}")
                skipped_tracks += 1
                continue
                
            if np.isnan(track_features).any():
                logger.warning(f"NaN values in features for track {i}")
                skipped_tracks += 1
                continue
            
            features.append(track_features)
            labels.append(label)
            label_counts[label] = label_counts.get(label, 0) + 1
            logger.debug(f"Successfully processed track {i} with label {label}")
            
        except Exception as e:
            logger.warning(f"Error processing track {i}: {e}")
            skipped_tracks += 1
            continue
    
    if not features:
        logger.error("No valid features extracted from any tracks")
        raise ValueError("No valid features extracted from the data")
    
    logger.info(f"Successfully processed {len(features)} tracks out of {total_tracks}")
    logger.info(f"Skipped {skipped_tracks} tracks")
    logger.info(f"Label distribution: {label_counts}")
    
    # Convert to numpy arrays
    features_array = np.array(features, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int64)
    
    logger.info(f"Feature array shape: {features_array.shape}")
    logger.info(f"Label array shape: {labels_array.shape}")
    
    # Balance the dataset
    features_array, labels_array = balance_dataset(features_array, labels_array)
    
    # Calculate class weights for imbalanced data (after balancing)
    unique_classes = np.unique(labels_array)
    class_counts = np.bincount(labels_array, minlength=4)  # Ensure we have weights for all 4 classes
    
    # Avoid division by zero by setting minimum count to 1
    class_counts = np.maximum(class_counts, 1)
    
    total_samples = len(labels_array)
    class_weights = torch.FloatTensor(total_samples / (len(unique_classes) * class_counts))
    
    logger.info(f"Final class counts: {class_counts}")
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    return features_array, labels_array, class_weights

def create_data_loaders(X, y, class_weights, batch_size=32, val_split=0.2):
    """Create train and validation data loaders with data augmentation"""
    
    # Check if we can do stratified split
    unique_labels, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)
    
    if min_count < 2:
        logger.warning(f"Cannot do stratified split - some classes have only {min_count} sample(s)")
        logger.warning("Falling back to random split")
        # Do random split instead of stratified
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, shuffle=True
        )
    else:
        # Split the data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Training class distribution: {Counter(y_train)}")
    logger.info(f"Validation class distribution: {Counter(y_val)}")
    
    # Create datasets
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, class_weights

def main():
    """Main training function"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create experiment
    experiment_name = "behavior_classification"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"Error with experiment: {e}")
        experiment_id = 0
    
    mlflow.set_experiment(experiment_name)
    
    try:
        # End any active runs
        if mlflow.active_run() is not None:
            mlflow.end_run()
        
        # Load data
        logger.info("Loading training data...")
        X, y, class_weights = load_training_data()
        
        # Get feature dimension
        feature_dim = X.shape[1]
        logger.info(f"Feature dimension: {feature_dim}")
        
        # Model configuration
        config = ModelConfig(
            input_size=feature_dim,
            hidden_size=0,      # No hidden layer
            num_classes=4,      # 4 behavior classes
            dropout=0.0,        # No dropout
            learning_rate=0.1,  # Higher learning rate
            epochs=200,         # More epochs
            batch_size=8,       # Smaller batch size
            weight_decay=0.0,   # No weight decay
            early_stopping_patience=20
        )
        
        logger.info(f"Dataset loaded: {len(X)} samples, {len(np.unique(y))} classes")
        
        # Create data loaders
        train_loader, val_loader, class_weights = create_data_loaders(X, y, class_weights, config.batch_size)
        
        # Create model
        model = BehaviorClassifier(
            input_size=config.input_size,
            num_classes=config.num_classes
        )
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(config.to_dict())
            
            # Train model
            logger.info("Starting model training...")
            trained_model, metrics = train_model(model, train_loader, val_loader, config, class_weights)
            
            # Log final metrics
            final_metrics = {
                'final_accuracy': metrics['final_accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'best_val_acc': metrics['best_val_acc']
            }
            mlflow.log_metrics(final_metrics)
            
            # Save classification report
            os.makedirs("outputs", exist_ok=True)
            report_path = "outputs/classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(metrics['classification_report'], f, indent=2)
            mlflow.log_artifact(report_path)
            
            # Save training history
            history_path = "outputs/training_history.json"
            with open(history_path, 'w') as f:
                json.dump(metrics['history'], f, indent=2)
            mlflow.log_artifact(history_path)
            
            # Create input example for model signature
            device = next(trained_model.parameters()).device
            input_example = torch.randn(1, feature_dim, device=device).cpu().numpy()
            
            # Log model with signature
            mlflow.pytorch.log_model(
                trained_model,
                "model",
                input_example=input_example,
                pip_requirements={
                    "torch": "2.0.0+cu118",
                    "torchvision": "0.15.0+cu118"
                }
            )
            
            # Save model locally
            os.makedirs("models", exist_ok=True)
            model_path = "models/behavior_classifier.pth"
            torch.save(trained_model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
            
            logger.info("Training completed successfully!")
            logger.info(f"Final accuracy: {metrics['final_accuracy']:.4f}")
            logger.info(f"F1 score: {metrics['f1_score']:.4f}")
            logger.info(f"Model saved to: {model_path}")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if mlflow.active_run() is not None:
            mlflow.end_run()
        raise
    finally:
        # Ensure we end any active runs
        if mlflow.active_run() is not None:
            mlflow.end_run()

if __name__ == "__main__":
    main()