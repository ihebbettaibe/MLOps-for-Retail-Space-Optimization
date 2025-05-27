# Retail Behavior Analysis System

A computer vision-based system for analyzing customer behavior in retail stores using CCTV footage.

## Project Structure
```
retail_analytics/
├── data/
│   ├── raw/              # Original DAV video files
│   ├── processed/        # Converted MP4 files
│   └── training/         # Training data and labels
├── models/
│   ├── weights/          # Model weights
│   └── checkpoints/      # Training checkpoints
├── src/
│   ├── data/            # Data processing scripts
│   ├── models/          # Model architecture
│   ├── utils/           # Utility functions
│   └── visualization/   # Visualization tools
├── notebooks/           # Jupyter notebooks for analysis
├── tests/              # Unit tests
└── results/            # Analysis results and reports
```

## Setup Instructions

1. **Environment Setup**:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. **Data Preparation**:
```bash
# Convert DAV files to MP4
python src/data/convert_videos.py --input_dir data/raw --output_dir data/processed

# Extract tracking data
python src/data/prepare_training_data.py --video_path data/processed/camera1.mp4 --output data/training/tracks.json
```

3. **Model Training**:
```bash
# Train behavior classifier
python src/models/train_behavior_classifier.py --data_path data/training/tracks.json --output_dir models/weights
```

4. **Run Analysis**:
```bash
# Analyze video
python src/analyze_video.py --video_path data/processed/camera1.mp4 --model_path models/weights/best_model.pth
```

## Usage

1. **Data Collection**:
   - Place your DAV video files in `data/raw/`
   - Run the conversion script to convert to MP4
   - Use the tracking script to extract customer trajectories

2. **Model Training**:
   - Label the extracted tracks with behavior categories
   - Train the behavior classifier
   - Monitor training with Weights & Biases

3. **Analysis**:
   - Run analysis on new videos
   - View heatmaps and behavior trends
   - Generate analytics reports

## Features

- Real-time customer tracking
- Behavior classification:
  - Hesitant
  - Decisive
  - Confused
  - Interested
  - Disinterested
  - Frustrated
- Heatmap generation
- Zone analysis
- Traffic flow visualization
- Behavior trend analysis

## Requirements

See `requirements.txt` for full list of dependencies.

## License

MIT License 