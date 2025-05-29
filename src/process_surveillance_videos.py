import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import logging
from datetime import datetime, timedelta
import re
from collections import defaultdict
import subprocess
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SurveillanceVideoProcessor:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.3):
        """
        Initialize the video processor with YOLO model.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for person detection
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0  # COCO dataset class ID for person
        
    def parse_filename(self, filename: str) -> Tuple[str, datetime]:
        """Extract channel number and timestamp from filename."""
        match = re.match(r'ch(\d+)_(\d{8})_(\d{4})\.mp4', filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        
        channel = f"ch{match.group(1)}"
        date_str = match.group(2)
        time_str = match.group(3)
        
        timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
        return channel, timestamp

    def detect_people(self, frame: np.ndarray) -> Tuple[bool, int]:
        """Detect people in a frame using YOLO."""
        results = self.model(frame, verbose=False)[0]
        person_detections = [box for box in results.boxes if 
                           int(box.cls) == self.person_class_id and 
                           box.conf > self.confidence_threshold]
        return len(person_detections) > 0, len(person_detections)

    def extract_clip(self, video_path: str, start_time: float, end_time: float, 
                    output_path: str) -> bool:
        """Extract a clip from the video using FFmpeg."""
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-y',  # Overwrite output file if exists
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting clip: {e.stderr.decode()}")
            return False

    def process_video(self, video_path: str, output_dir: str, 
                     buffer_seconds: int = 5, 
                     min_gap_seconds: int = 600) -> List[Dict]:
        """
        Process a single video file and extract clips with human activity.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save clips
            buffer_seconds: Seconds to include before/after activity
            min_gap_seconds: Minimum gap between clips (10 minutes)
        
        Returns:
            List of dictionaries containing clip information
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        clips_info = []
        current_clip = None
        last_detection_time = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = frame_count / fps
            has_people, num_people = self.detect_people(frame)
            
            if has_people:
                if current_clip is None:
                    # Start new clip
                    current_clip = {
                        'start_time': max(0, current_time - buffer_seconds),
                        'end_time': current_time + buffer_seconds,
                        'num_people': num_people
                    }
                else:
                    # Extend current clip
                    current_clip['end_time'] = current_time + buffer_seconds
                    current_clip['num_people'] = max(current_clip['num_people'], num_people)
                
                last_detection_time = current_time
            
            elif current_clip is not None:
                # Check if we should end the current clip
                if current_time - last_detection_time > min_gap_seconds:
                    # Save the clip
                    clip_filename = f"clip_{len(clips_info):04d}.mp4"
                    output_path = os.path.join(output_dir, clip_filename)
                    
                    if self.extract_clip(video_path, current_clip['start_time'], 
                                       current_clip['end_time'], output_path):
                        clips_info.append({
                            'filename': clip_filename,
                            'start_time': current_clip['start_time'],
                            'end_time': current_clip['end_time'],
                            'num_people': current_clip['num_people']
                        })
                    
                    current_clip = None
            
            frame_count += 1
            
            # Log progress every 1000 frames
            if frame_count % 1000 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        return clips_info

    def process_directory(self, input_dir: str, output_base_dir: str):
        """Process all videos in a directory and organize by channel."""
        # Group files by channel
        channel_files = defaultdict(list)
        for file in os.listdir(input_dir):
            if file.endswith('.mp4'):
                try:
                    channel, timestamp = self.parse_filename(file)
                    channel_files[channel].append((file, timestamp))
                except ValueError as e:
                    logger.warning(f"Skipping {file}: {e}")
        
        # Process each channel
        for channel, files in channel_files.items():
            logger.info(f"Processing channel {channel} with {len(files)} files")
            
            # Create channel output directory
            channel_dir = os.path.join(output_base_dir, f"{channel}_clips")
            os.makedirs(channel_dir, exist_ok=True)
            
            # Process files in chronological order
            files.sort(key=lambda x: x[1])
            all_clips_info = []
            
            for filename, timestamp in files:
                video_path = os.path.join(input_dir, filename)
                logger.info(f"Processing {filename}")
                
                clips_info = self.process_video(video_path, channel_dir)
                
                # Add timestamp information to clips
                for clip in clips_info:
                    clip['original_video'] = filename
                    clip['original_timestamp'] = timestamp
                    clip['clip_start_datetime'] = timestamp + timedelta(seconds=clip['start_time'])
                    clip['clip_end_datetime'] = timestamp + timedelta(seconds=clip['end_time'])
                
                all_clips_info.extend(clips_info)
            
            # Save clip information to CSV
            if all_clips_info:
                df = pd.DataFrame(all_clips_info)
                csv_path = os.path.join(channel_dir, f"{channel}_clips_info.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved clip information to {csv_path}")

def main():
    # Initialize processor
    processor = SurveillanceVideoProcessor()
    
    # Process videos
    input_dir = "datap2m"  # Directory containing input videos
    output_dir = "results/surveillance_clips"  # Directory for output clips
    
    os.makedirs(output_dir, exist_ok=True)
    processor.process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main() 