import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import re
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from ultralytics import YOLO
import ffmpeg
from tqdm import tqdm
import shutil
import tempfile
import gc
import psutil
import time

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, input_dir='datap2m', output_dir='processed_clips', model_path='models/yolov8n.pt'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.model = None
        self.clip_data = defaultdict(list)
        self.clip_counters = defaultdict(int)
        self.processed_files = set()  # Track processed files
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO model
        logger.info("Loading YOLO model...")
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.min_detection_frames = 5  # Minimum consecutive frames with detection
        self.max_gap_frames = 1800  # 1 minute at 30fps
        
        # Clip parameters
        self.buffer_seconds = 5  # Seconds to keep before detection
        self.clip_padding = 5  # Seconds to keep after last detection
        
        # Processing parameters
        self.process_resolution = (640, 360)  # Width, Height for processing
        self.buffer_resolution = (854, 480)   # Width, Height for buffer storage
        self.save_resolution = (1280, 720)    # Width, Height for saved clips
        self.base_frame_skip = 3  # Base frame skip rate
        self.max_frame_skip = 6   # Maximum frame skip rate
        self.memory_threshold = 85  # Memory usage threshold (percentage)
        
        # Load previously processed files
        self.load_processed_files()
        
        logger.info(f"Initialized VideoProcessor with:")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model path: {self.model_path}")
        
    def load_processed_files(self):
        """Load list of previously processed files"""
        processed_file = self.output_dir / 'processed_files.txt'
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                self.processed_files = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.processed_files)} previously processed files")
    
    def save_processed_files(self):
        """Save list of processed files"""
        processed_file = self.output_dir / 'processed_files.txt'
        with open(processed_file, 'w') as f:
            for file in self.processed_files:
                f.write(f"{file}\n")
        logger.info(f"Saved {len(self.processed_files)} processed files to {processed_file}")

    def convert_dav_to_mp4(self, input_file: Path, output_file: Path) -> bool:
        """Convert DAV file to MP4 using ffmpeg"""
        try:
            # Use ffmpeg to convert the file
            stream = ffmpeg.input(str(input_file))
            stream = ffmpeg.output(stream, str(output_file))
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            return True
        except ffmpeg.Error as e:
            logger.error(f"Error converting {input_file}: {e.stderr.decode()}")
            return False
            
    def extract_channel_info(self, filename: str) -> tuple:
        """Extract channel number and timestamp from filename"""
        # Pattern: XVR_ch{number}_main_{timestamp1}_{timestamp2}.dav
        pattern = r'XVR_ch(\d+)_main_(\d{14})_(\d{14})\.dav'
        match = re.match(pattern, filename)
        if match:
            channel = int(match.group(1))
            start_time = datetime.strptime(match.group(2), '%Y%m%d%H%M%S')
            end_time = datetime.strptime(match.group(3), '%Y%m%d%H%M%S')
            return channel, start_time, end_time
        return None, None, None

    def save_clip_from_frames(self, frames: list, output_path: Path, fps: float):
        """Save a clip from a list of frames using ffmpeg."""
        try:
            logger.info(f"Starting to save clip using ffmpeg...")
            logger.info(f"Number of frames to save: {len(frames)}")
            logger.info(f"Target FPS: {fps}")
            logger.info(f"Output path: {output_path}")
            
            # Get frame dimensions from the first frame
            height, width = frames[0].shape[:2]
            logger.info(f"Frame dimensions: {width}x{height}")
            
            # Create temporary directory for frame files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Save frames as temporary image files
                frame_files = []
                for i, frame in enumerate(frames):
                    frame_path = temp_dir_path / f"frame_{i:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_files.append(str(frame_path))
                
                # Create ffmpeg process using image sequence
                try:
                    stream = (
                        ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
                        .output(str(output_path), 
                               pix_fmt='yuv420p',
                               vcodec='libx264',
                               preset='ultrafast',  # Faster encoding
                               crf=23,  # Good quality, reasonable file size
                               r=fps)
                        .overwrite_output()
                        .global_args('-y')  # Overwrite output file
                    )
                    
                    # Run ffmpeg with explicit error handling
                    process = stream.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
                    
                    # Write frames to ffmpeg process
                    for frame in frames:
                        process.stdin.write(frame.tobytes())
                    
                    # Close stdin and wait for process to complete
                    process.stdin.close()
                    stdout, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        error_msg = stderr.decode('utf-8') if stderr else "No error message available"
                        logger.error(f"FFmpeg error details:\n{error_msg}")
                        raise ffmpeg.Error('ffmpeg', stdout, stderr)
                    
                    # Verify the output file exists and has content
                    if not output_path.exists():
                        raise Exception("Output file was not created")
                    
                    file_size = output_path.stat().st_size
                    if file_size == 0:
                        raise Exception("Output file is empty")
                    
                    logger.info(f"Successfully saved clip: {output_path.name} ({file_size/1024/1024:.1f} MB)")
                    
                except ffmpeg.Error as e:
                    logger.error(f"FFmpeg error occurred: {str(e)}")
                    if e.stderr:
                        logger.error(f"FFmpeg stderr output:\n{e.stderr.decode('utf-8')}")
                    raise
                except Exception as e:
                    logger.error(f"Error during ffmpeg processing: {str(e)}")
                    raise
                finally:
                    # Clean up temporary files
                    for frame_file in frame_files:
                        try:
                            Path(frame_file).unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Failed to delete temporary file {frame_file}: {e}")
                
        except Exception as e:
            logger.error(f"Unexpected error while saving clip: {str(e)}")
            # Clean up output file if it exists but is incomplete
            if output_path.exists():
                try:
                    output_path.unlink()
                    logger.info(f"Cleaned up incomplete output file: {output_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up incomplete output: {cleanup_error}")
            raise

    def get_memory_usage(self):
        """Get current memory usage percentage"""
        return psutil.Process().memory_percent()

    def adjust_frame_skip(self, current_skip):
        """Adjust frame skip rate based on memory usage"""
        memory_usage = self.get_memory_usage()
        if memory_usage > self.memory_threshold:
            return min(current_skip + 1, self.max_frame_skip)
        elif memory_usage < self.memory_threshold - 10:
            return max(current_skip - 1, self.base_frame_skip)
        return current_skip

    def resize_frame(self, frame, target_size):
        """Resize frame while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Calculate aspect ratios
        aspect_ratio = w / h
        target_ratio = target_w / target_h
        
        if aspect_ratio > target_ratio:
            # Width is larger relative to height
            new_w = target_w
            new_h = int(new_w / aspect_ratio)
        else:
            # Height is larger relative to width
            new_h = target_h
            new_w = int(new_h * aspect_ratio)
            
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def process_video(self, video_path: Path, channel: int):
        """Process a single video file to extract clips with human activity"""
        logger.info(f"Starting to process video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {fps} FPS, {total_frames} frames, {duration:.2f} seconds")
        
        # Initialize variables
        frame_count = 0
        detection_start = None
        last_detection_frame = None
        in_detection = False
        frames_to_save = []
        buffer_frames = []
        current_frame_skip = 1  # Reduced frame skip for better detection
        
        # Create progress bar
        pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame for buffer storage (using smaller resolution)
                buffer_frame = self.resize_frame(frame, (640, 360))  # Reduced resolution for buffer
                
                # Add frame to buffer (keep only last N frames)
                buffer_frames.append(buffer_frame)
                if len(buffer_frames) > int(self.buffer_seconds * fps):
                    buffer_frames.pop(0)
                    
                # Resize frame for processing (using smaller resolution)
                process_frame = self.resize_frame(frame, (640, 360))  # Reduced resolution for processing
                
                # Detect people
                results = self.model(process_frame, classes=[0], conf=self.confidence_threshold)  # class 0 is person
                has_detection = len(results[0].boxes) > 0
                
                if has_detection:
                    if not in_detection:
                        # Start of new detection - include buffer frames
                        frames_to_save = buffer_frames.copy()
                        detection_start = frame_count - len(buffer_frames)
                        in_detection = True
                    else:
                        frames_to_save.append(buffer_frame)
                    last_detection_frame = frame_count
                elif in_detection:
                    # Check if we should end the current clip
                    if frame_count - last_detection_frame > self.max_gap_frames:
                        # Add padding frames after last detection
                        padding_frames = int(self.clip_padding * fps)
                        for _ in range(padding_frames):
                            ret, frame = cap.read()
                            if ret:
                                buffer_frame = self.resize_frame(frame, (640, 360))  # Reduced resolution
                                frames_to_save.append(buffer_frame)
                                frame_count += 1
                                pbar.update(1)
                        
                        # Save clip if it's long enough
                        if len(frames_to_save) >= self.min_detection_frames:
                            try:
                                # Process frames in smaller batches to save memory
                                batch_size = 30  # Process 30 frames at a time
                                save_frames = []
                                for i in range(0, len(frames_to_save), batch_size):
                                    batch = frames_to_save[i:i + batch_size]
                                    resized_batch = [self.resize_frame(f, self.save_resolution) for f in batch]
                                    save_frames.extend(resized_batch)
                                    # Clear batch from memory
                                    del resized_batch
                                    gc.collect()
                                
                                self.save_clip(save_frames, channel, video_path, 
                                             detection_start, last_detection_frame, fps)
                                # Clear frames from memory
                                del save_frames
                                gc.collect()
                            except Exception as e:
                                logger.error(f"Error saving clip: {str(e)}")
                                # Clear frames from memory
                                gc.collect()
                        
                        # Reset for next clip
                        frames_to_save = []
                        in_detection = False
                        detection_start = None
                        
                        # Force garbage collection
                        gc.collect()
                    else:
                        frames_to_save.append(buffer_frame)
                
                frame_count += 1
                pbar.update(1)
                
                # Periodically force garbage collection
                if frame_count % 30 == 0:  # More frequent garbage collection
                    gc.collect()
                
        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}", exc_info=True)
        finally:
            # Save final clip if needed
            if in_detection and len(frames_to_save) >= self.min_detection_frames:
                try:
                    # Process frames in smaller batches
                    batch_size = 30
                    save_frames = []
                    for i in range(0, len(frames_to_save), batch_size):
                        batch = frames_to_save[i:i + batch_size]
                        resized_batch = [self.resize_frame(f, self.save_resolution) for f in batch]
                        save_frames.extend(resized_batch)
                        del resized_batch
                        gc.collect()
                    
                    self.save_clip(save_frames, channel, video_path,
                                 detection_start, last_detection_frame, fps)
                    del save_frames
                except Exception as e:
                    logger.error(f"Error saving final clip: {str(e)}")
            
            pbar.close()
            cap.release()
            
            # Final garbage collection
            gc.collect()
            
            # Add logging for detection statistics
            logger.info(f"Finished processing {video_path.name}")
            logger.info(f"Total frames processed: {frame_count}")
            logger.info(f"Clips saved for channel {channel}: {self.clip_counters[channel]}")

    def save_clip(self, frames: list, channel: int, video_path: Path,
                 start_frame: int, end_frame: int, fps: float):
        """Save a clip and update clip data"""
        try:
            # Generate clip filename
            clip_num = self.clip_counters[channel] + 1
            clip_filename = f"ch{channel}_clip_{clip_num:04d}.mp4"
            clip_path = self.output_dir / f"ch{channel}_clips" / clip_filename
            
            # Create channel directory if it doesn't exist
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory for clips: {clip_path.parent}")
            
            logger.info(f"Attempting to save clip {clip_filename} ({len(frames)} frames)")
            logger.info(f"Full path: {clip_path.absolute()}")
            
            # Calculate timestamps
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            # Get original video timestamps
            _, video_start_time, _ = self.extract_channel_info(video_path.name)
            if video_start_time:
                clip_start_time = video_start_time + timedelta(seconds=start_time)
                clip_end_time = video_start_time + timedelta(seconds=end_time)
                logger.info(f"Clip timestamps: {clip_start_time} to {clip_end_time}")
            else:
                clip_start_time = None
                clip_end_time = None
                logger.warning("Could not determine clip timestamps")
            
            # Save clip using ffmpeg
            logger.info("Starting to save clip using ffmpeg...")
            self.save_clip_from_frames(frames, clip_path, fps)
            
            # Verify the file was created
            if clip_path.exists():
                file_size = clip_path.stat().st_size / (1024*1024)  # Size in MB
                logger.info(f"Successfully saved clip: {clip_filename} ({file_size:.1f} MB)")
            else:
                logger.error(f"Clip file was not created: {clip_path}")
                return
            
            # Update clip data
            clip_info = {
                'clip_filename': clip_filename,
                'original_video': video_path.name,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': clip_start_time,
                'end_time': clip_end_time,
                'duration': end_time - start_time,
                'num_frames': len(frames)
            }
            self.clip_data[channel].append(clip_info)
            logger.info(f"Updated clip data for channel {channel}")
            
            # Update counter
            self.clip_counters[channel] = clip_num
            logger.info(f"Updated clip counter for channel {channel} to {clip_num}")
            
        except Exception as e:
            logger.error(f"Error saving clip {clip_filename}: {str(e)}", exc_info=True)
            # Try to clean up if the file was partially created
            if clip_path.exists():
                try:
                    clip_path.unlink()
                    logger.info(f"Cleaned up partial clip file: {clip_filename}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up partial clip: {cleanup_error}")

    def save_clip_reports(self):
        """Save CSV reports for each channel"""
        logger.info("\nSaving clip reports...")
        for channel, data in self.clip_data.items():
            if not data:
                logger.info(f"No clips to report for channel {channel}")
                continue
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Save to CSV
            report_path = self.output_dir / f"ch{channel}_clips" / f"ch{channel}_report.csv"
            logger.info(f"Saving report for channel {channel} to: {report_path}")
            
            try:
                df.to_csv(report_path, index=False)
                if report_path.exists():
                    logger.info(f"Successfully saved report: {report_path}")
                else:
                    logger.error(f"Report file was not created: {report_path}")
            except Exception as e:
                logger.error(f"Error saving report for channel {channel}: {str(e)}", exc_info=True)

    def process_all_videos(self):
        """Process all videos in the input directory"""
        logger.info("Starting to process all videos")
        
        # Group files by channel
        channel_files = defaultdict(list)
        total_files = 0
        skipped_files = 0
        
        for file in self.input_dir.glob('*.dav'):
            if str(file) in self.processed_files:
                logger.info(f"Skipping already processed file: {file.name}")
                skipped_files += 1
                continue
                
            channel, start_time, end_time = self.extract_channel_info(file.name)
            if channel is not None:
                channel_files[channel].append(file)
                total_files += 1
                logger.info(f"Found video file for channel {channel}: {file.name}")
                logger.info(f"  Duration: {end_time - start_time}")
                logger.info(f"  Size: {file.stat().st_size / (1024*1024):.1f} MB")
        
        if not channel_files:
            if skipped_files > 0:
                logger.info(f"All {skipped_files} files have been processed previously!")
            else:
                logger.warning("No .dav files found in input directory!")
            return
            
        logger.info(f"Found {total_files} new videos across {len(channel_files)} channels")
        logger.info(f"Skipped {skipped_files} previously processed files")
        
        # Process each channel's videos
        for channel, files in channel_files.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing channel {channel} ({len(files)} files)")
            logger.info(f"{'='*50}\n")
            
            # Sort files by timestamp
            files.sort(key=lambda x: self.extract_channel_info(x.name)[1])
            
            for i, file in enumerate(files, 1):
                try:
                    logger.info(f"\nProcessing file {i}/{len(files)} for channel {channel}")
                    logger.info(f"File: {file.name}")
                    
                    # Convert DAV to MP4 if needed
                    mp4_path = file.with_suffix('.mp4')
                    if not mp4_path.exists():
                        logger.info(f"Converting {file.name} to MP4...")
                        if not self.convert_dav_to_mp4(file, mp4_path):
                            logger.error(f"Failed to convert {file.name}")
                            continue
                        logger.info("Conversion completed successfully")
                    else:
                        logger.info(f"MP4 file already exists: {mp4_path.name}")
                    
                    # Process the video
                    logger.info(f"Starting video processing: {mp4_path.name}")
                    self.process_video(mp4_path, channel)
                    logger.info(f"Completed processing: {mp4_path.name}")
                    
                    # Mark file as processed
                    self.processed_files.add(str(file))
                    self.save_processed_files()  # Save after each file
                    
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)
                    continue
                
        # Save reports
        logger.info("\nSaving clip reports...")
        self.save_clip_reports()
        logger.info("Processing complete!")
        
        # Print summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total channels processed: {len(channel_files)}")
        for channel, files in channel_files.items():
            logger.info(f"Channel {channel}:")
            logger.info(f"  Files processed: {len(files)}")
            logger.info(f"  Clips extracted: {self.clip_counters[channel]}")
            if self.clip_data[channel]:
                total_duration = sum(clip['duration'] for clip in self.clip_data[channel])
                logger.info(f"  Total clip duration: {total_duration:.1f} seconds")

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('video_processing.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Process only datatest.mp4
    test_file = processor.input_dir / 'datatest.mp4'
    test_channel = 1  # Using channel 1 for the test file
    
    if test_file.exists():
        logger.info(f"\nProcessing test video:")
        logger.info(f"Test file: {test_file.name}")
        logger.info(f"Channel: {test_channel}")
        
        # Process the video
        processor.process_video(test_file, test_channel)
        processor.save_clip_reports()
        logger.info("\nTest processing complete!")
    else:
        logger.error(f"Test video file not found: {test_file}")

if __name__ == '__main__':
    main() 