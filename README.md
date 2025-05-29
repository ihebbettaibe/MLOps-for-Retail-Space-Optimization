# MLOps for Retail Space Optimization

## Academic Project Information
**Tunisian Republic**  
Ministry of Higher Education and Scientific Research  
University of Carthage  
Higher School of Communications of Tunis  

**PROJECT REPORT P2M**  
MLOps for Retail Space Optimization  

**By**  
Iheb Ben Taieb  
Zeineb Louati  

**Supervised By**  
Asma Ben Letaifa  

**Academic Year**: 2024 - 2025

## Project Overview
A computer vision-based system for analyzing customer behavior in retail stores using CCTV footage. This system includes behavior classification, heatmap generation, and comprehensive analytics.

## Features

- Real-time customer tracking and detection using YOLOv8
- Behavior classification:
  - Hesitant
  - Decisive
  - Confused
  - Interested
  - Disinterested
  - Frustrated
- Heatmap generation for customer movement patterns
- Zone analysis and traffic flow visualization
- Behavior trend analysis
- MLflow integration for experiment tracking
- Interactive dashboard using Streamlit

## System Architecture

### Overview
Our system combines computer vision models, behavior analysis algorithms, and automation workflows to analyze retail store footage and extract meaningful data about customer movements and behaviors.

### Architecture Diagrams
[Placeholder for architecture diagrams]
- System Overview Diagram
- Use Case Diagram
- Class Diagram
- Sequence Diagram

### Key Components
1. **Video Processing Pipeline**
   - Multi-camera video capture
   - YOLOv8 person detection
   - Movement tracking and analysis

2. **Behavior Analysis**
   - LSTM-GRU hybrid model
   - Real-time behavior classification
   - Pattern recognition

3. **MLOps Integration**
   - MLflow experiment tracking
   - Model versioning
   - Performance monitoring

4. **Automation Workflow**
   - n8n workflow automation
   - Automated data processing
   - Real-time dashboard updates

## Dashboard Screenshots
[Placeholder for dashboard screenshots]
- Main Dashboard View
- Heatmap Visualization
- Behavior Analytics
- MLflow Integration Interface

## Project Structure
```
p2m/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── src/                    # All source code here
│   ├── __init__.py
│   ├── heatmap.py
│   ├── train_behavior_classifier.py
│   ├── run_mlflow.py
│   ├── dashboard.py
│   ├── behavior_detector.py
│   ├── mlops_pipeline.py
│   ├── prepare_training_data.py
│   └── config.yaml
│
├── tests/                  # All test code here
│   └── (test files)
│
├── notebooks/              # Jupyter notebooks
│   └── (notebook files)
│
├── models/                 # Model weights/checkpoints (ignored by git)
├── data/                   # Data (ignored by git)
├── datap2m/                # Video data (ignored by git)
├── model_registry/         # Model registry (ignored by git)
│
└── scripts/                # Utility scripts
    └── setup_directories.ps1
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

2. **Running the Heatmap Analysis**:
```bash
# Generate heatmap from video
python src/heatmap.py
```

3. **Running the Behavior Analysis**:
```bash
# Train behavior classifier
python src/train_behavior_classifier.py

# Run behavior detection
python src/behavior_detector.py
```

4. **Viewing the Dashboard**:
```bash
# Launch the Streamlit dashboard
streamlit run src/dashboard.py
```

## Technologies Used

- **Core Technologies**:
  - Python
  - YOLOv8
  - PyTorch
  - OpenCV
  - FFmpeg

- **MLOps & Monitoring**:
  - MLflow
  - Weights & Biases
  - n8n

- **Web Interface**:
  - Streamlit
  - Plotly

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 