import cv2
from ultralytics import solutions
import os

# Create output directory if it doesn't exist
os.makedirs("analysis_results", exist_ok=True)

# Input video path
video_path = "datap2m/datatest.mp4"
assert os.path.exists(video_path), f"Video file not found: {video_path}"

# Output paths
output_video = "analysis_results/heatmap_output.mp4"
output_data = "analysis_results/heatmap_data.json"

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Use default YOLO model that will be downloaded automatically
heatmap = solutions.Heatmap(
    show=True,  
    model="yolov8n.pt",  # This will download the model automatically
    colormap=cv2.COLORMAP_PARULA,
)

# Store results for each frame
frame_results = []

# Process video
frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = heatmap(im0)
    
    # Save frame results
    frame_data = {
        "frame": frame_count,
        "timestamp": frame_count / fps,
        "results": results
    }
    frame_results.append(frame_data)
    
    # Write frame to output video
    video_writer.write(results.plot_im)
    
    frame_count += 1

# Save results to JSON
import json
with open(output_data, 'w') as f:
    json.dump({
        "video_info": {
            "width": w,
            "height": h,
            "fps": fps,
            "total_frames": frame_count
        },
        "frame_results": frame_results
    }, f, indent=2)

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Processing complete!")
print(f"Output video saved to: {output_video}")
print(f"Analysis data saved to: {output_data}")  