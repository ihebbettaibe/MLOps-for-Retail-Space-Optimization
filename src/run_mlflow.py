import subprocess
import os
import webbrowser
import time
from pathlib import Path
import socket
import sys
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_mlflow_ui(port=5000):
    """Run MLflow UI and open in browser"""
    try:
        # Create mlruns directory if it doesn't exist
        Path("mlruns").mkdir(exist_ok=True)
        
        # Check if port is already in use
        if is_port_in_use(port):
            print(f"Port {port} is already in use. Trying port {port + 1}")
            port += 1
            if is_port_in_use(port):
                print(f"Port {port} is also in use. Please free up a port or specify a different one.")
                sys.exit(1)
        
        # Start MLflow UI
        mlflow_ui = subprocess.Popen(
            ["mlflow", "ui", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(2)
        
        # Check if process is still running
        if mlflow_ui.poll() is not None:
            stdout, stderr = mlflow_ui.communicate()
            print("Error starting MLflow UI:")
            print(stderr)
            sys.exit(1)
        
        # Open browser
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        
        print(f"MLflow UI is running at {url}")
        print("Press Ctrl+C to stop the server")
        
        try:
            mlflow_ui.wait()
        except KeyboardInterrupt:
            print("\nStopping MLflow UI server...")
            mlflow_ui.terminate()
            mlflow_ui.wait()
            print("MLflow UI server stopped")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

def log_metrics(self, detections, processing_time, total_frames):
    """Log metrics to MLflow for behavior analysis"""
    try:
        import mlflow
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # Initialize MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        print("MLflow tracking URI set to:", mlflow.get_tracking_uri())
        
        # Create or get experiment
        experiment_name = "behavior_analysis"
        try:
            print(f"Looking for experiment: {experiment_name}")
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                print(f"Creating new experiment: {experiment_name}")
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created experiment with ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                print(f"Found existing experiment with ID: {experiment_id}")
                
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            print(f"Set current experiment to: {experiment_name}")
            
        except Exception as e:
            print(f"Error with experiment creation: {e}")
            experiment_id = 0
        
        # Start a new run
        print("Starting new MLflow run...")
        with mlflow.start_run(run_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            print("MLflow run started successfully")
            
            # Calculate basic metrics
            total_detections = len(detections)
            if total_detections == 0:
                # Log baseline metrics if no detections
                mlflow.log_metrics({
                    'total_detections': 0,
                    'avg_confidence': 0.0,
                    'processing_time': processing_time,
                    'detection_rate': 0.0,
                    'avg_dwell_time': 0.0,
                    'behavior_ratio': 0.0,
                    'tracking_quality': 0.0
                })
                return
                
            # Calculate behavior statistics
            interested_count = sum(1 for d in detections if d['behavior'] == 'Interested')
            not_interested_count = sum(1 for d in detections if d['behavior'] == 'Not Interested')
            
            # Calculate average confidence
            avg_confidence = np.mean([d['confidence'] for d in detections])
            
            # Calculate detection rate (detections per second)
            detection_rate = total_detections / (processing_time if processing_time > 0 else 1)
            
            # Calculate behavior ratio (interested vs not interested)
            behavior_ratio = interested_count / total_detections if total_detections > 0 else 0
            
            # Calculate average dwell time
            avg_dwell_time = np.mean([d['dwell_time'] for d in detections])
            
            # Calculate tracking quality (based on confidence and consistency)
            tracking_quality = avg_confidence * (1 - np.std([d['confidence'] for d in detections]))
            
            # Get predictions and true labels for classification metrics
            predictions = [1 if d['behavior'] == 'Interested' else 0 for d in detections]
            true_labels = [1 if d['confidence'] > 0.5 else 0 for d in detections]  # Using confidence as proxy for true labels
            
            # Calculate classification metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted', zero_division=0)
            
            # Log metrics to MLflow
            metrics = {
                # Basic metrics
                'total_detections': total_detections,
                'avg_confidence': avg_confidence,
                'processing_time': processing_time,
                'detection_rate': detection_rate,
                'avg_dwell_time': avg_dwell_time,
                'behavior_ratio': behavior_ratio,
                'tracking_quality': tracking_quality,
                
                # Classification metrics from behavior classifier
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                
                # Additional metrics
                'interested_count': interested_count,
                'not_interested_count': not_interested_count,
                'max_dwell_time': max([d['dwell_time'] for d in detections]),
                'min_dwell_time': min([d['dwell_time'] for d in detections]),
                'detection_consistency': 1 - np.std([d['confidence'] for d in detections])
            }
            
            print("Logging metrics:", metrics)
            mlflow.log_metrics(metrics)
            
            # Log parameters
            params = {
                'total_frames': total_frames,
                'fps': total_frames / processing_time if processing_time > 0 else 0,
                'detection_threshold': 0.5
            }
            print("Logging parameters:", params)
            mlflow.log_params(params)
            
            print("Metrics and parameters logged successfully to MLflow")
            
    except ImportError:
        print("MLflow not installed. Skipping metrics logging.")
    except Exception as e:
        print(f"Error logging metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Start MLflow UI
    run_mlflow_ui()
    
    # Wait for MLflow UI to start
    print("Waiting for MLflow UI to start...")
    time.sleep(5)  # Increased wait time
    
    # Example usage of log_metrics
    try:
        # Create a dummy detector instance
        class DummyDetector:
            def log_metrics(self, detections, processing_time, total_frames):
                from run_mlflow import log_metrics
                log_metrics(self, detections, processing_time, total_frames)
        
        detector = DummyDetector()
        
        # Example detections
        detections = [
            {'behavior': 'Interested', 'confidence': 0.8, 'dwell_time': 2.5},
            {'behavior': 'Not Interested', 'confidence': 0.3, 'dwell_time': 1.0},
            {'behavior': 'Interested', 'confidence': 0.9, 'dwell_time': 3.0}
        ]
        
        print("\nStarting metrics logging...")
        # Log metrics
        detector.log_metrics(detections, processing_time=10.0, total_frames=300)
        print("\nMetrics logging completed. Check MLflow UI for results.")
        
    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()