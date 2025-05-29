import cv2
import json
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict
import argparse
from tqdm import tqdm
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataCollector:
    """Collect and prepare training data from video files"""
    
    def __init__(
        self,
        video_path: str,
        output_path: str,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        min_track_length: int = 30  # Minimum frames for a valid track
    ):
        """
        Initialize the data collector
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the collected data
            model_path: Path to the YOLO model
            confidence_threshold: Detection confidence threshold
            min_track_length: Minimum number of frames for a valid track
        """
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.confidence_threshold = confidence_threshold
        self.min_track_length = min_track_length
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize YOLO detector
        self.detector = YOLO(model_path)
        
        # Initialize tracking
        self.tracks = {}
        self.track_history = defaultdict(list)
        self.track_timestamps = defaultdict(list)
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def process_video(self):
        """Process the video and collect tracking data"""
        logger.info(f"Processing video: {self.video_path}")
        logger.info(f"Total frames: {self.total_frames}")
        
        frame_idx = 0
        pbar = tqdm(total=self.total_frames, desc="Processing frames")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Get current timestamp
            timestamp = frame_idx / self.fps
            
            # Run detection
            results = self.detector.track(
                frame,
                persist=True,
                conf=self.confidence_threshold,
                classes=[0]  # Only track people (class 0)
            )
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    # Get center point
                    center_x, center_y = box[0], box[1]
                    
                    # Update track history
                    self.track_history[track_id].append((center_x, center_y))
                    self.track_timestamps[track_id].append(timestamp)
            
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        self.cap.release()
        
        # Filter and save tracks
        self._filter_and_save_tracks()
        
    def _filter_and_save_tracks(self):
        """Filter tracks and save to JSON"""
        filtered_tracks = []
        
        for track_id in self.track_history:
            positions = self.track_history[track_id]
            timestamps = self.track_timestamps[track_id]
            
            # Filter short tracks
            if len(positions) < self.min_track_length:
                continue
                
            # Convert positions to list of dicts
            position_dicts = [
                {"x": float(x), "y": float(y)}
                for x, y in positions
            ]
            
            # Create track entry
            track = {
                "track_id": int(track_id),
                "positions": position_dicts,
                "timestamps": [float(t) for t in timestamps],
                "behavior": None  # To be labeled manually
            }
            
            filtered_tracks.append(track)
        
        # Save to JSON
        output_data = {
            "metadata": {
                "video_path": str(self.video_path),
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "total_frames": self.total_frames,
                "num_tracks": len(filtered_tracks),
                "created_at": datetime.now().isoformat()
            },
            "tracks": filtered_tracks
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Saved {len(filtered_tracks)} tracks to {self.output_path}")
        
    def visualize_tracks(self, output_video_path: Optional[str] = None):
        """
        Visualize tracks on the video
        
        Args:
            output_video_path: Path to save the visualization video
        """
        if output_video_path is None:
            output_video_path = self.output_path.with_suffix('.mp4')
            
        # Reopen video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        frame_idx = 0
        pbar = tqdm(total=self.total_frames, desc="Visualizing tracks")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_idx / self.fps
            
            # Draw tracks
            for track_id, positions in self.track_history.items():
                if len(positions) < self.min_track_length:
                    continue
                    
                # Find the closest position to current timestamp
                timestamps = self.track_timestamps[track_id]
                if not timestamps:
                    continue
                    
                idx = np.searchsorted(timestamps, timestamp)
                if idx >= len(positions):
                    continue
                    
                x, y = positions[idx]
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                
                # Draw track history
                for i in range(1, len(positions)):
                    pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                    pt2 = (int(positions[i][0]), int(positions[i][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # Write frame
            out.write(frame)
            
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        out.release()
        
        logger.info(f"Saved visualization to {output_video_path}")

def main():
    # Set up paths
    video_path = "datap2m/datatest.mp4"
    output_path = "data/training_data.json"  # Changed to save directly in data folder
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create data collector
    collector = TrainingDataCollector(
        video_path=video_path,
        output_path=output_path,
        confidence_threshold=0.5,
        min_track_length=15  # Reduced for shorter video
    )
    
    try:
        # Process video and collect tracks
        collector.process_video()
        logger.info(f"Training data saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 