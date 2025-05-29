import cv2
import numpy as np
from ultralytics import YOLO, solutions
import time
from collections import defaultdict, deque
import logging
import os
from datetime import datetime
import torch
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.feature_size = 128  # Size of feature vector
        
    def extract_features(self, frame=None, bbox=None, positions=None, timestamps=None):
        """
        Extract features from either a frame+bbox or a sequence of positions+timestamps
        
        Args:
            frame: Input frame (optional)
            bbox: Bounding box coordinates (optional)
            positions: List of (x,y) positions (optional)
            timestamps: List of timestamps (optional)
        """
        if frame is not None and bbox is not None:
            return self._extract_frame_features(frame, bbox)
        elif positions is not None and timestamps is not None:
            return self._extract_sequence_features(positions, timestamps)
        else:
            raise ValueError("Must provide either (frame, bbox) or (positions, timestamps)")
            
    def _extract_frame_features(self, frame, bbox):
        """Extract features from a bounding box in the frame"""
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(self.feature_size)
            
        # Resize ROI to fixed size
        roi = cv2.resize(roi, (64, 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate HOG features
        hog = cv2.HOGDescriptor()
        features = hog.compute(gray)
        
        return features.flatten()
        
    def _extract_sequence_features(self, positions, timestamps):
        """Extract features from a sequence of positions and timestamps"""
        if len(positions) < 2:
            return np.zeros(8)  # Return zero features for short sequences
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(positions)):
            try:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                dt = timestamps[i] - timestamps[i-1]
                
                if dt > 0:  # Avoid division by zero
                    vx = dx / dt
                    vy = dy / dt
                    velocities.append([vx, vy])
            except (IndexError, TypeError) as e:
                logger.warning(f"Error calculating velocity: {e}")
                continue
        
        if not velocities:
            return np.zeros(8)  # Return zero features if no valid velocities
        
        velocities = np.array(velocities)
        
        # Calculate velocity statistics
        mean_vel = np.mean(velocities, axis=0)  # 2 features
        std_vel = np.std(velocities, axis=0)    # 2 features
        max_vel = np.max(velocities, axis=0)    # 2 features
        min_vel = np.min(velocities, axis=0)    # 2 features
        
        # Combine features (total 8 features)
        features = np.concatenate([
            mean_vel,    # 2 features
            std_vel,     # 2 features
            max_vel,     # 2 features
            min_vel      # 2 features
        ])
        
        return features

class BehaviorClassifier(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_classes=2, num_layers=2, dropout=0.3, bidirectional=True):
        super(BehaviorClassifier, self).__init__()
        
        # Store input size as instance variable
        self.input_size = input_size
        
        # Fully connected layers for 2D input
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, input_size)
        return self.fc(x)
        
    def predict(self, features):
        """Predict behavior class from features"""
        with torch.no_grad():
            # Ensure features are in the right shape (batch_size, input_size)
            if len(features.shape) == 1:
                features = features.unsqueeze(0)  # Add batch dimension
                
            features = torch.FloatTensor(features)
            outputs = self(features)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.numpy()

class BehaviorDetector:
    def __init__(self, video_path):
        """Initialize the behavior detector with a video path"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize YOLO detector and heatmap
        self.detector = YOLO("yolov8n.pt")
        self.heatmap = solutions.Heatmap(
            show=True,
            model="yolov8n.pt",
            colormap=cv2.COLORMAP_PARULA,
            line_width=2
        )
        
        # Tracking parameters
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # 1 second at 30fps
        self.track_states = defaultdict(lambda: {
            'position': None,
            'velocity': None,
            'dwell_time': 0,
            'last_update': 0,
            'behavior': 'Unknown',
            'confidence': 0.0
        })
        
        # Behavior detection parameters
        self.movement_threshold = 8.0  # pixels per frame
        self.stillness_threshold = 15  # frames
        self.interest_time_threshold = 3.0  # seconds
        self.velocity_window = 5       # frames for velocity calculation
        
        # Track states
        self.track_last_positions = {}
        self.track_stillness_count = defaultdict(int)
        self.track_viewing_time = defaultdict(float)
        self.track_speeds = defaultdict(list)
        
        # Visualization parameters
        self.colors = {
            'Interested': (0, 255, 0),    # Green
            'Not Interested': (0, 0, 255), # Red
            'Unknown': (128, 128, 128)    # Gray
        }
        
        # Create output directory
        self.output_dir = "analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def calculate_speed(self, track_id, current_position, current_time):
        """Calculate movement speed"""
        if track_id in self.track_last_positions:
            last_pos = self.track_last_positions[track_id]
            last_time = self.track_viewing_time.get(track_id, current_time)
            
            if current_time > last_time:
                distance = np.sqrt(
                    (current_position[0] - last_pos[0])**2 + 
                    (current_position[1] - last_pos[1])**2
                )
                speed = distance / (current_time - last_time)
                self.track_speeds[track_id].append(speed)
                return speed
        return 0.0

    def update_behavior(self, track_id, current_pos, current_time):
        """Update behavior state based on movement and viewing time"""
        state = self.track_states[track_id]
        
        # Calculate speed
        speed = self.calculate_speed(track_id, current_pos, current_time)
        
        # Update viewing time based on movement
        if speed < self.movement_threshold:
            state['dwell_time'] += 1/self.fps
            self.track_stillness_count[track_id] += 1
        else:
            # Gradually decrease viewing time when moving
            state['dwell_time'] = max(0, state['dwell_time'] - 0.5/self.fps)
            self.track_stillness_count[track_id] = 0
        
        # Multi-factor behavior determination
        viewing_time = state['dwell_time']
        stillness = self.track_stillness_count[track_id]
        
        # Determine behavior based on multiple factors
        is_interested = (
            viewing_time > self.interest_time_threshold or
            stillness >= self.stillness_threshold or
            (speed < self.movement_threshold * 0.5 and viewing_time > 1.0)
        )
        
        # Update behavior and confidence
        if is_interested:
            state['behavior'] = 'Interested'
            # Higher confidence for longer viewing time and stillness
            state['confidence'] = min(1.0, (viewing_time / self.interest_time_threshold) * 0.7 + 
                                    (stillness / self.stillness_threshold) * 0.3)
        else:
            state['behavior'] = 'Not Interested'
            state['confidence'] = min(1.0, 1.0 - (viewing_time / self.interest_time_threshold))
        
        state['position'] = current_pos
        state['last_update'] = current_time
        
        return state['behavior'], state['confidence']
        
    def process_frame(self, frame, frame_idx):
        """Process a single frame and return detection data"""
        current_time = frame_idx / self.fps
        
        # Process frame with Ultralytics heatmap
        results = self.heatmap(frame)
        
        # Get the heatmap visualization
        frame_with_heatmap = results.plot_im
        
        # Detect people for behavior analysis
        detections = self.detector(frame, classes=[0], conf=0.3)
        
        frame_detections = []
        
        # Process detections
        for result in detections:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Update behavior
                    behavior, confidence = self.update_behavior(i, center, current_time)
                    
                    # Add detection data
                    detection = {
                        'frame': frame_idx,
                        'timestamp': current_time,
                        'track_id': i,
                        'behavior': behavior,
                        'confidence': confidence,
                        'x': center[0],
                        'y': center[1],
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'dwell_time': self.track_states[i]['dwell_time']
                    }
                    frame_detections.append(detection)
                    
                    # Draw bounding box and label on the heatmap frame
                    color = self.colors[behavior]
                    cv2.rectangle(frame_with_heatmap, (x1, y1), (x2, y2), color, 2)
                    
                    # Add behavior label with confidence
                    label = f"{behavior} ({confidence:.2f})"
                    cv2.putText(frame_with_heatmap, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_detections
        
    def get_heatmap_overlay(self, frame_shape, alpha=0.5):
        """Generate heatmap overlay for the frame"""
        # Create a blank heatmap
        heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        
        # Add heat for each track
        for track_id, positions in self.track_history.items():
            for pos in positions:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                    # Add Gaussian heat at each position
                    cv2.circle(heatmap, (x, y), 20, 1.0, -1)
        
        # Normalize heatmap
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_colored
        
    def analyze_video(self):
        """Analyze video and return detection results"""
        if not self.video_path:
            raise ValueError("No video path provided")
        
        # Initialize video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join("analysis_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video writer with H.264 codec
        output_video_path = os.path.join(output_dir, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Fallback to XVID if H.264 is not available
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_video_path = os.path.join(output_dir, "output_video.avi")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise ValueError("Could not create output video writer")
        
        # Initialize results
        detections = []
        frame_idx = 0
        start_time = time.time()
        
        # Process video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with YOLO detector
            results = self.detector(frame, classes=[0], conf=0.3)
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            
            # Process detections
            frame_detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = map(int, box)
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        # Update behavior
                        behavior, confidence = self.update_behavior(i, center, frame_idx / fps)
                        
                        # Add detection data
                        detection = {
                            'frame': frame_idx,
                            'timestamp': frame_idx / fps,
                            'track_id': i,
                            'behavior': behavior,
                            'confidence': confidence,
                            'x': center[0],
                            'y': center[1],
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'dwell_time': self.track_states[i]['dwell_time']
                        }
                        frame_detections.append(detection)
                        
                        # Draw bounding box and label
                        color = self.colors[behavior]
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{behavior} ({confidence:.2f})"
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if frame_detections:
                detections.extend(frame_detections)
            
            # Write frame to output video
            out.write(display_frame)
            
            frame_idx += 1
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clean up
        cap.release()
        out.release()
        
        # Return results
        return {
            'detections': detections,
            'processing_time': processing_time,
            'total_frames': frame_idx,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'output_video': output_video_path,
            'output_dir': output_dir
        }

if __name__ == "__main__":
    # Configuration
    video_path = "datap2m/datatest.mp4"
    output_path = "analysis_results/analyzed_video.mp4"
    
    try:
        # Initialize detector
        detector = BehaviorDetector(video_path)
        
        # Run analysis
        results = detector.analyze_video()
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved in: {os.path.dirname(video_path)}")
        print("Generated files:")
        print(f"- Full analysis video: {video_path}")
        print(f"- Heatmap-only video: {os.path.join(os.path.dirname(video_path), 'heatmap_only.mp4')}")
        print(f"- Final heatmap image: {os.path.join(os.path.dirname(video_path), 'final_heatmap.png')}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")
        print("Please check the video file and try again.")