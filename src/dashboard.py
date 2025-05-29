import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import cv2
from PIL import Image
import io
import time
import base64
from streamlit_option_menu import option_menu
import tempfile
import shutil
import torch

# Initialize session state
if 'pipeline' not in st.session_state:
    try:
        from mlops_pipeline import EnhancedMLOpsPipeline
        st.session_state.pipeline = EnhancedMLOpsPipeline()
    except Exception as e:
        st.error(f"Error initializing pipeline: {e}")
        st.session_state.pipeline = None

if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None
if 'heatmap_data' not in st.session_state:
    st.session_state.heatmap_data = np.zeros((480, 640), dtype=np.float32)
if 'video_frame' not in st.session_state:
    st.session_state.video_frame = 0
if 'video_playing' not in st.session_state:
    st.session_state.video_playing = False
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Overview"
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None

# Video path
VIDEO_PATH = r"C:\Users\ASUS\Desktop\p2m\analysis_results\analyzed_video.mp4"

def load_metrics_history():
    """Load metrics history from file"""
    try:
        with open(os.path.join(st.session_state.pipeline.data_dir, "metrics_history.json"), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_video_info(video_path):
    """Get video information"""
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }

def get_video_frame(video_path, frame_number):
    """Extract a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    return None

def run_behavior_detection(video_path):
    """Run behavior detection on uploaded video"""
    # Initialize variables
    output_dir = None
    results = None
    
    try:
        # Create output directory first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("analysis_results", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Import and initialize detector in a separate try block
        try:
            import behavior_detector
            detector = behavior_detector.BehaviorDetector(video_path=video_path)
        except ImportError as e:
            st.error(f"Failed to import behavior_detector: {str(e)}")
            return None, None
        except Exception as e:
            st.error(f"Failed to initialize detector: {str(e)}")
            return None, None
        
        # Process video in a separate try block
        try:
            results = detector.analyze_video()
            if results is None:
                st.error("No results returned from video analysis")
                return None, None
                
            # Store results in session state
            st.session_state.analysis_results = results
            
            # Update the processed video path in session state
            if 'output_video' in results:
                st.session_state.processed_video_path = results['output_video']
            
        except Exception as e:
            st.error(f"Error during video analysis: {str(e)}")
            return None, None
            
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None, None
        
    finally:
        # Clean up if we have no results
        if output_dir and os.path.exists(output_dir) and results is None:
            try:
                shutil.rmtree(output_dir)
                output_dir = None
            except Exception as cleanup_error:
                st.warning(f"Could not clean up temporary directory: {str(cleanup_error)}")
    
    return results, output_dir

def create_behavior_timeline_chart(detections):
    """Create a timeline chart showing behavior detections over time"""
    if not detections:
        return None
    
    # Convert detections to DataFrame
    df_data = []
    for i, detection in enumerate(detections):
        df_data.append({
            'frame': i,
            'timestamp': detection.get('timestamp', f"Frame {i}"),
            'behavior': detection.get('behavior', 'Unknown'),
            'confidence': detection.get('confidence', 0),
            'x': detection.get('x', 0),
            'y': detection.get('y', 0)
        })
    
    df = pd.DataFrame(df_data)
    
    # Create timeline scatter plot
    fig = px.scatter(
        df,
        x='frame',
        y='behavior',
        color='behavior',
        size='confidence',
        hover_data=['timestamp', 'confidence', 'x', 'y'],
        title='Behavior Detection Timeline',
        labels={'frame': 'Frame Number', 'behavior': 'Detected Behavior'}
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_confidence_distribution(detections):
    """Create a histogram of confidence scores"""
    if not detections:
        return None
    
    confidences = [d.get('confidence', 0) for d in detections]
    
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title='Confidence Score Distribution',
        labels={'x': 'Confidence Score', 'y': 'Frequency'},
        template='plotly_dark'
    )
    
    fig.update_layout(height=300)
    return fig

def create_behavior_statistics(detections):
    """Create behavior statistics and charts"""
    if not detections:
        return None, None
    
    # Count behaviors
    behavior_counts = {}
    for detection in detections:
        behavior = detection.get('behavior', 'Unknown')
        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
    
    # Create pie chart
    fig_pie = px.pie(
        values=list(behavior_counts.values()),
        names=list(behavior_counts.keys()),
        title='Behavior Distribution',
        template='plotly_dark'
    )
    
    # Create bar chart
    fig_bar = px.bar(
        x=list(behavior_counts.keys()),
        y=list(behavior_counts.values()),
        title='Behavior Frequency',
        labels={'x': 'Behavior Type', 'y': 'Count'},
        template='plotly_dark'
    )
    
    return fig_pie, fig_bar

def display_detection_results(results, output_dir):
    """Display comprehensive detection results"""
    if not results:
        st.error("No results to display")
        return
    
    # Main metrics
    st.markdown("""
        <div style='background: linear-gradient(135deg, #2E2E2E 0%, #3E3E3E 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;'>
            <h2 style='color: #4ECDC4; margin-bottom: 1rem;'>🎯 Detection Results</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Detections", 
            len(results.get('detections', [])),
            delta=f"+{len(results.get('detections', []))}"
        )
    
    with col2:
        st.metric(
            "Processing Time", 
            f"{results.get('processing_time', 0):.2f}s",
            delta=None
        )
    
    with col3:
        avg_confidence = np.mean([d.get('confidence', 0) for d in results.get('detections', [])]) if results.get('detections') else 0
        st.metric(
            "Avg Confidence", 
            f"{avg_confidence:.1%}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Frames Processed", 
            results.get('total_frames', 0),
            delta=None
        )
    
    # Video and detection visualization
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📹 Processed Video")
        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
            st.video(st.session_state.processed_video_path)
        
        # Show output video if available
        output_video_path = os.path.join(output_dir, "output_video.mp4")
        if os.path.exists(output_video_path):
            st.subheader("🎬 Analyzed Video (with detections)")
            st.video(output_video_path)
    
    with col2:
        st.subheader("🔥 Activity Heatmap")
        heatmap_path = os.path.join(output_dir, "heatmap.png")
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Activity Heatmap", use_container_width=True)
        else:
            st.info("Heatmap not generated")
    
    # Detection details
    if results.get('detections'):
        detections = results['detections']
        
        # Behavior statistics
        st.subheader("📊 Behavior Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie, fig_bar = create_behavior_statistics(detections)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Timeline visualization
        st.subheader("⏱️ Detection Timeline")
        timeline_fig = create_behavior_timeline_chart(detections)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Confidence distribution
        st.subheader("📈 Confidence Distribution")
        confidence_fig = create_confidence_distribution(detections)
        if confidence_fig:
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        # Detection details table
        st.subheader("📋 Detection Details")
        
        # Convert detections to DataFrame
        detection_data = []
        for i, detection in enumerate(detections):
            detection_data.append({
                'Frame': i,
                'Timestamp': detection.get('timestamp', f"Frame {i}"),
                'Behavior': detection.get('behavior', 'Unknown'),
                'Confidence': f"{detection.get('confidence', 0):.1%}",
                'X': detection.get('x', 0),
                'Y': detection.get('y', 0),
                'Width': detection.get('width', 0),
                'Height': detection.get('height', 0)
            })
        
        df = pd.DataFrame(detection_data)
        
        # Display with pagination
        page_size = 20
        total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
        
        if total_pages > 1:
            page = st.selectbox("Select Page", range(1, total_pages + 1))
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True, hide_index=True)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download results
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download detection data as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Detection Data (CSV)",
                data=csv,
                file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download results as JSON
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📄 Download Full Results (JSON)",
                data=json_data,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.info("No detections found in the video")

def show_live_analysis():
    """Display enhanced live analysis page with file upload and behavior detection"""
    st.markdown("""
        <div style='background: linear-gradient(135deg, #2E2E2E 0%, #3E3E3E 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;'>
            <h2 style='color: #4ECDC4; margin-bottom: 1rem;'>🔴 Live Analysis</h2>
            <p style='color: #CCCCCC; margin: 0;'>Upload an MP4 video file to analyze behavior patterns using AI detection</p>
        </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose an MP4 file",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video file to analyze behavior patterns"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.1f} MB)")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Get video info
            video_info = get_video_info(temp_path)
            
            if video_info:
                # Display video info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Duration", f"{video_info['duration']:.1f}s")
                with col2:
                    st.metric("FPS", f"{video_info['fps']:.1f}")
                with col3:
                    st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
                with col4:
                    st.metric("Total Frames", f"{video_info['frame_count']:,}")
                
                # Process button
                if st.button("🚀 Start Behavior Analysis", use_container_width=True, type="primary"):
                    # Run behavior detection
                    with st.spinner("🔍 Analyzing video... This may take a few minutes depending on video length."):
                        results, output_dir = run_behavior_detection(temp_path)
                        
                        if results and output_dir:
                            st.success("✅ Analysis completed successfully!")
                            
                            # Display results immediately
                            st.markdown("### 📊 Analysis Results")
                            
                            # Create two columns for video and metrics
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Display analyzed video
                                if 'output_video' in results and os.path.exists(results['output_video']):
                                    st.markdown("#### 🎥 Processed Video with Detections")
                                    video_file = open(results['output_video'], 'rb')
                                    video_bytes = video_file.read()
                                    st.video(video_bytes)
                                else:
                                    st.error("Analyzed video not found. Please check the output directory.")
                            
                            with col2:
                                # Display detection metrics
                                if 'detections' in results:
                                    st.markdown("#### 🎯 Detection Metrics")
                                    st.metric("Total Detections", len(results['detections']))
                                    avg_conf = sum(d.get('confidence', 0) for d in results['detections']) / len(results['detections'])
                                    st.metric("Average Confidence", f"{avg_conf:.1%}")
                                    st.metric("Processing Time", f"{results.get('processing_time', 0):.2f}s")
                            
                            # Display detection history
                            if 'detections' in results:
                                st.markdown("### 📈 Detection History")
                                detection_data = []
                                for i, detection in enumerate(results['detections']):
                                    detection_data.append({
                                        'Frame': i,
                                        'Behavior': detection.get('behavior', 'Unknown'),
                                        'Confidence': f"{detection.get('confidence', 0):.1%}",
                                        'Location': f"({detection.get('x', 0)}, {detection.get('y', 0)})"
                                    })
                                st.dataframe(pd.DataFrame(detection_data))
                                
                                # Display behavior distribution
                                st.markdown("### 📊 Behavior Distribution")
                                behavior_counts = {}
                                for detection in results['detections']:
                                    behavior = detection.get('behavior', 'Unknown')
                                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                                
                                fig = px.pie(
                                    values=list(behavior_counts.values()),
                                    names=list(behavior_counts.keys()),
                                    title='Behavior Distribution'
                                )
                                st.plotly_chart(fig)
                            
                            # Download options
                            st.markdown("### 💾 Download Results")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("📥 Download Detection Data (CSV)"):
                                    df = pd.DataFrame(results['detections'])
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name="detection_results.csv",
                                        mime="text/csv"
                                    )
                            with col2:
                                if st.button("📥 Download Full Results (JSON)"):
                                    json_str = json.dumps(results, indent=2)
                                    st.download_button(
                                        label="Download JSON",
                                        data=json_str,
                                        file_name="analysis_results.json",
                                        mime="application/json"
                                    )
                        else:
                            st.error("❌ Analysis failed. Please check the error messages above.")
            
            else:
                st.error("❌ Could not read video file. Please ensure it's a valid video format.")
        
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a video file to begin analysis")

def plot_metrics_history(metrics_history):
    """Create plots from metrics history"""
    if not metrics_history:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'timestamp': datetime.fromisoformat(m['timestamp']),
            'accuracy': m['metrics']['classification_report']['accuracy'],
            'precision': m['metrics']['classification_report']['weighted avg']['precision'],
            'recall': m['metrics']['metrics']['classification_report']['weighted avg']['recall'],
            'f1-score': m['metrics']['classification_report']['weighted avg']['f1-score']
        }
        for m in metrics_history
    ])
    
    # Create line plot
    fig = go.Figure()
    for metric in ['accuracy', 'precision', 'recall', 'f1-score']:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[metric],
            name=metric.capitalize(),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Model Performance Metrics Over Time',
        xaxis_title='Timestamp',
        yaxis_title='Score',
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig

def plot_confusion_matrix(metrics_history):
    """Plot latest confusion matrix"""
    if not metrics_history:
        return None
    
    latest_cm = metrics_history[-1]['metrics']['confusion_matrix']
    fig = px.imshow(
        latest_cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Not Interested', 'Interested'],
        y=['Not Interested', 'Interested'],
        color_continuous_scale='Viridis',
        template='plotly_dark'
    )
    
    fig.update_layout(title='Latest Confusion Matrix')
    return fig

def show_overview():
    """Display overview dashboard"""
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("🎯 System Status")
        
        # System metrics in a more organized layout
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Active Detections", "12", delta="2")
            st.metric("Model Version", st.session_state.pipeline.current_model_version if st.session_state.pipeline else "v1.0")
        with metric_col2:
            st.metric("Processing Time", "45ms", delta="-5ms")
            st.metric("Uptime", "2h 15m")
        
        # System health gauge with better styling
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=85,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "System Health", 'font': {'size': 16}},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#FF6B6B'},
                    {'range': [50, 80], 'color': '#FFE66D'},
                    {'range': [80, 100], 'color': '#4ECDC4'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent Activity with icons
        st.subheader("📊 Recent Activity")
        activity_container = st.container()
        with activity_container:
            current_time = datetime.now()
            activities = [
                ("🎯", "New detection", 0),
                ("🔄", "Behavior change", 1),
                ("⚡", "Processing frame", 2),
                ("✅", "Model updated", 5),
                ("🔧", "System check", 10)
            ]
            
            for icon, activity, minutes_ago in activities:
                time_str = (current_time - timedelta(minutes=minutes_ago)).strftime('%H:%M:%S')
                st.markdown(f"**{icon} {time_str}** - {activity}")
    
    with col2:
        st.subheader("🎬 Video Analysis")
        
        # Video player section
        if os.path.exists(VIDEO_PATH):
            video_info = get_video_info(VIDEO_PATH)
            
            if video_info:
                # Video information
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Duration", f"{video_info['duration']:.1f}s")
                with info_col2:
                    st.metric("FPS", f"{video_info['fps']:.1f}")
                with info_col3:
                    st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
                
                # Frame slider
                st.subheader("Frame Analysis")
                frame_slider = st.slider(
                    "Select Frame", 
                    0, 
                    video_info['frame_count']-1, 
                    st.session_state.video_frame
                )
                
                if frame_slider != st.session_state.video_frame:
                    st.session_state.video_frame = frame_slider
                
                frame = get_video_frame(VIDEO_PATH, st.session_state.video_frame)
                if frame is not None:
                    st.image(frame, caption=f"Frame {st.session_state.video_frame}", use_container_width=True)
            else:
                st.error("Could not load video information")
        else:
            st.error(f"Video file not found: {VIDEO_PATH}")
            st.info("Please ensure the analyzed video exists at the specified path.")
        
        # Activity Heatmap
        st.subheader("🔥 Activity Heatmap")
        heatmap_path = "analysis_results/heatmap_20250414_135926.png"
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Activity Heatmap", use_container_width=True)
        else:
            st.warning("Heatmap image not found. Upload a video in Live Analysis to generate heatmaps.")

def show_model_performance():
    """Display model performance metrics"""
    st.markdown("""
        <div style='background: linear-gradient(135deg, #2E2E2E 0%, #3E3E3E 100%); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;'>
            <h2 style='color: #4ECDC4; margin-bottom: 1rem;'>📈 Model Performance Analytics</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Create sample metrics data (replace with actual metrics in production)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)  # 4 days including today
    sample_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sample_data = {
        'timestamp': sample_dates,
        'accuracy': np.random.normal(0.85, 0.05, len(sample_dates)).clip(0.7, 0.95),
        'precision': np.random.normal(0.82, 0.05, len(sample_dates)).clip(0.7, 0.95),
        'recall': np.random.normal(0.80, 0.05, len(sample_dates)).clip(0.7, 0.95),
        'f1_score': np.random.normal(0.81, 0.05, len(sample_dates)).clip(0.7, 0.95),
        'inference_time': np.random.normal(0.15, 0.02, len(sample_dates)).clip(0.1, 0.2),
        'confidence': np.random.normal(0.88, 0.03, len(sample_dates)).clip(0.8, 0.95)
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(sample_data)
    
    # Create tabs for different metric views
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🎯 Detection Analysis", "⚡ System Metrics"])
    
    with tab1:
        # Main metrics over time
        st.subheader("Performance Metrics Over Time")
        
        # Create separate plots for each metric
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy plot
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['accuracy'],
                name='Accuracy',
                mode='lines+markers',
                line=dict(width=3, color='#4ECDC4'),
                marker=dict(size=8)
            ))
            fig_acc.update_layout(
                title='Accuracy Over Time',
                xaxis_title='Date',
                yaxis_title='Score',
                hovermode='x unified',
                template='plotly_dark',
                yaxis=dict(range=[0.7, 1.0], tickformat='.2%'),
                height=400
            )
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # Precision plot
            fig_prec = go.Figure()
            fig_prec.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['precision'],
                name='Precision',
                mode='lines+markers',
                line=dict(width=3, color='#FF6B6B'),
                marker=dict(size=8)
            ))
            fig_prec.update_layout(
                title='Precision Over Time',
                xaxis_title='Date',
                yaxis_title='Score',
                hovermode='x unified',
                template='plotly_dark',
                yaxis=dict(range=[0.7, 1.0], tickformat='.2%'),
                height=400
            )
            st.plotly_chart(fig_prec, use_container_width=True)
        
        with col2:
            # Recall plot
            fig_rec = go.Figure()
            fig_rec.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['recall'],
                name='Recall',
                mode='lines+markers',
                line=dict(width=3, color='#FFE66D'),
                marker=dict(size=8)
            ))
            fig_rec.update_layout(
                title='Recall Over Time',
                xaxis_title='Date',
                yaxis_title='Score',
                hovermode='x unified',
                template='plotly_dark',
                yaxis=dict(range=[0.7, 1.0], tickformat='.2%'),
                height=400
            )
            st.plotly_chart(fig_rec, use_container_width=True)
            
            # F1 Score plot
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['f1_score'],
                name='F1 Score',
                mode='lines+markers',
                line=dict(width=3, color='#95E1D3'),
                marker=dict(size=8)
            ))
            fig_f1.update_layout(
                title='F1 Score Over Time',
                xaxis_title='Date',
                yaxis_title='Score',
                hovermode='x unified',
                template='plotly_dark',
                yaxis=dict(range=[0.7, 1.0], tickformat='.2%'),
                height=400
            )
            st.plotly_chart(fig_f1, use_container_width=True)
        
        # Current metrics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{df['accuracy'].iloc[-1]:.1%}", 
                     f"{df['accuracy'].iloc[-1] - df['accuracy'].iloc[-2]:.1%}")
        with col2:
            st.metric("Precision", f"{df['precision'].iloc[-1]:.1%}", 
                     f"{df['precision'].iloc[-1] - df['precision'].iloc[-2]:.1%}")
        with col3:
            st.metric("Recall", f"{df['recall'].iloc[-1]:.1%}", 
                     f"{df['recall'].iloc[-1] - df['recall'].iloc[-2]:.1%}")
        with col4:
            st.metric("F1 Score", f"{df['f1_score'].iloc[-1]:.1%}", 
                     f"{df['f1_score'].iloc[-1] - df['f1_score'].iloc[-2]:.1%}")
    
    with tab2:
        # Detection analysis
        st.subheader("Detection Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            cm = np.array([[85, 15], [12, 88]])
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Not Interested', 'Interested'],
                y=['Not Interested', 'Interested'],
                color_continuous_scale='Viridis',
                template='plotly_dark',
                title='Confusion Matrix'
            )
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    fig_cm.add_annotation(
                        x=j, y=i,
                        text=str(cm[i][j]),
                        showarrow=False,
                        font=dict(color='white', size=14)
                    )
            
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # ROC curve
            fpr = np.linspace(0, 1, 100)
            tpr = 1 - np.exp(-5 * fpr)  # Simulated ROC curve
            auc = 0.92
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC curve (AUC = {auc:.2f})',
                mode='lines',
                line=dict(width=3, color='#4ECDC4')
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                mode='lines',
                line=dict(dash='dash', color='red')
            ))
            
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                template='plotly_dark',
                showlegend=True
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Detection distribution
        st.subheader("Detection Distribution")
        detection_data = {
            'Behavior': ['Interested', 'Not Interested', 'Unknown'],
            'Count': [120, 85, 15],
            'Confidence': [0.88, 0.82, 0.65]
        }
        df_det = pd.DataFrame(detection_data)
        
        fig_dist = px.bar(
            df_det,
            x='Behavior',
            y='Count',
            color='Confidence',
            color_continuous_scale='Viridis',
            template='plotly_dark',
            title='Detection Distribution by Behavior'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab3:
        # System metrics
        st.subheader("System Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Inference time over time
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['inference_time'],
                mode='lines+markers',
                name='Inference Time',
                line=dict(width=3, color='#4ECDC4')
            ))
            
            fig_time.update_layout(
                title='Inference Time Over Time',
                xaxis_title='Date',
                yaxis_title='Time (seconds)',
                template='plotly_dark'
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig_conf = go.Figure()
            fig_conf.add_trace(go.Histogram(
                x=df['confidence'],
                nbinsx=20,
                name='Confidence',
                marker_color='#4ECDC4'
            ))
            
            fig_conf.update_layout(
                title='Confidence Score Distribution',
                xaxis_title='Confidence Score',
                yaxis_title='Frequency',
                template='plotly_dark'
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # System metrics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Inference Time", f"{df['inference_time'].mean():.3f}s", 
                     f"{df['inference_time'].iloc[-1] - df['inference_time'].mean():.3f}s")
        with col2:
            st.metric("Avg Confidence", f"{df['confidence'].mean():.1%}", 
                     f"{df['confidence'].iloc[-1] - df['confidence'].mean():.1%}")
        with col3:
            st.metric("Total Detections", "1,234", "+123")
        with col4:
            st.metric("Model Version", "v2.1.0", "Updated")

def main():
    st.set_page_config(
        page_title="🎯 Behavior Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🎯"
    )
    
    # Enhanced CSS styling
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
            color: white;
        }
        .stMetric {
            background: linear-gradient(135deg, #2E2E2E 0%, #3E3E3E 100%);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #4E4E4E;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }
        .stMetric:hover {
            transform: translateY(-5px);
        }
        .stMetric > div {
            background-color: transparent !important;
        }
        .stMetric label {
            color: #CCCCCC !important;
            font-size: 1.1em;
        }
        .stMetric div[data-testid="stMetricValue"] {
            font-size: 1.8em;
            font-weight: bold;
        }
        .stSidebar {
            background: linear-gradient(180deg, #2E2E2E 0%, #1E1E1E 100%);
            padding: 2rem 1rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: bold;
            padding: 0.8rem 1.5rem;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            background: linear-gradient(135deg, #44A08D 0%, #4ECDC4 100%);
        }
        h1, h2, h3 {
            color: #4ECDC4 !important;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }
        .stSelectbox > div > div {
            background-color: #3E3E3E;
            border-color: #4E4E4E;
            border-radius: 10px;
            padding: 0.5rem;
        }
        .stProgress > div > div {
            background-color: #4ECDC4;
        }
        .stProgress > div > div > div {
            background-color: #44A08D;
        }
        .stAlert {
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .stDataFrame {
            background-color: #2E2E2E;
            border-radius: 12px;
            padding: 1rem;
        }
        .stPlotlyChart {
            background-color: #2E2E2E;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with enhanced styling
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #4ECDC4, #44A08D); margin-bottom: 2rem; border-radius: 15px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);">
            <h1 style="color: white; margin: 0; font-size: 2.8em; font-weight: 700;">🎯 Behavior Detection System</h1>
            <p style="color: white; margin: 1rem 0 0 0; font-size: 1.3em; opacity: 0.9;">Advanced AI-Powered Analysis Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar with modern navigation
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #4ECDC4; font-size: 1.8em;">🚀 Navigation</h2>
            </div>
        """, unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["Overview", "Model Performance", "Live Analysis"],
            icons=['house', 'graph-up', 'camera-video'],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#2E2E2E"},
                "icon": {"color": "#4ECDC4", "font-size": "1.2rem"},
                "nav-link": {
                    "font-size": "1.1rem",
                    "text-align": "left",
                    "margin": "0.5rem 0",
                    "padding": "0.8rem 1rem",
                    "border-radius": "10px",
                    "color": "white",
                    "background-color": "#3E3E3E",
                },
                "nav-link-selected": {
                    "background-color": "#4ECDC4",
                    "color": "white",
                    "font-weight": "bold",
                },
            }
        )
        
        st.session_state.selected_page = selected
        
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; margin: 1rem 0;">
                <h3 style="color: #4ECDC4; font-size: 1.4em;">📊 Quick Stats</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Enhanced metrics display
        col1, col2 = st.columns(2)
        with col1:
            st.metric("System Status", "🟢 Online", delta="Stable")
        with col2:
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"), delta="Just now")
    
    # Page routing with enhanced layouts
    if st.session_state.selected_page == "Overview":
        show_overview()
    elif st.session_state.selected_page == "Model Performance":
        show_model_performance()
    elif st.session_state.selected_page == "Live Analysis":
        show_live_analysis()
    
    # Enhanced footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888888; padding: 1.5rem; background: linear-gradient(90deg, #2E2E2E, #3E3E3E); border-radius: 10px;'>
            <p style='margin: 0; font-size: 1.1em;'>
                Behavior Detection System v2.0 | Powered by AI | 
                <span style='color: #4ECDC4;'>Last updated: {}</span>
            </p>
        </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()