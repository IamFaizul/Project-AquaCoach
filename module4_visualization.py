"""
MODULE 4: VISUALIZATION
Create annotated videos with skeleton overlay and angle display
"""

import cv2
import json
import numpy as np
from pathlib import Path
import mediapipe as mp


# Colors (BGR format)
COLORS = {
    'gold': (0, 215, 255),      # Gold/Orange for high quality
    'good': (0, 255, 0),        # Green for good quality
    'medium': (0, 165, 255),    # Orange for medium quality
    'poor': (0, 0, 255),        # Red for poor quality
    'text': (255, 255, 255),    # White for text
    'background': (0, 0, 0),    # Black for text background
}


# Landmark connections (MediaPipe skeleton)
CONNECTIONS = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]


def create_annotated_video(video_path, pose_data_path, quality_report_path, metrics_path, output_folder='output'):
    """Create annotated video with skeleton overlay"""
    
    print(f"\n{'='*60}")
    print(f"CREATING ANNOTATED VIDEO")
    print(f"{'='*60}\n")
    
    # Load data
    with open(pose_data_path, 'r') as f:
        pose_data = json.load(f)
    
    with open(quality_report_path, 'r') as f:
        quality_data = json.load(f)
    
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    video_name = Path(video_path).stem
    frames = pose_data['frames']
    gold_frames = set(quality_data['validation']['gold_frame_indices'])
    frame_metrics = {m['frame_number']: m for m in metrics_data['frame_metrics']}
    
    print(f"Video: {video_name}")
    print(f"Gold frames: {len(gold_frames)}\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    Path(output_folder).mkdir(exist_ok=True)
    output_path = Path(output_folder) / f"{video_name}_annotated.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_num = 0
    
    print("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get frame data
        frame_data = frames[frame_num] if frame_num < len(frames) else None
        is_gold = frame_num in gold_frames
        metrics = frame_metrics.get(frame_num)
        
        # Annotate frame
        annotated = annotate_frame(frame, frame_data, is_gold, metrics, width, height)
        
        out.write(annotated)
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  {frame_num}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"\n✓ SAVED: {output_path}\n")
    
    return str(output_path)


def annotate_frame(frame, frame_data, is_gold, metrics, width, height):
    """Annotate a single frame"""
    
    annotated = frame.copy()
    
    if not frame_data or not frame_data['landmarks']:
        # No detection - add red border
        cv2.rectangle(annotated, (0, 0), (width-1, height-1), COLORS['poor'], 10)
        add_text(annotated, "NO DETECTION", (20, 50), COLORS['poor'])
        return annotated
    
    landmarks = frame_data['landmarks']
    
    # Determine frame quality color
    if is_gold:
        color = COLORS['gold']
        quality_text = "GOLD FRAME"
    else:
        # Check visibility
        avg_visibility = np.mean([lm['visibility'] for lm in landmarks.values()])
        if avg_visibility > 0.7:
            color = COLORS['good']
            quality_text = "GOOD"
        elif avg_visibility > 0.5:
            color = COLORS['medium']
            quality_text = "MEDIUM"
        else:
            color = COLORS['poor']
            quality_text = "LOW QUALITY"
    
    # Draw border
    cv2.rectangle(annotated, (0, 0), (width-1, height-1), color, 8)
    
    # Draw skeleton
    draw_skeleton(annotated, landmarks, width, height, color)
    
    # Draw angles if available
    if metrics:
        draw_angles(annotated, landmarks, metrics, width, height)
    
    # Add quality text
    add_text(annotated, quality_text, (20, 50), color, size=1.2)
    
    return annotated


def draw_skeleton(frame, landmarks, width, height, color):
    """Draw skeleton on frame"""
    
    # Convert normalized coordinates to pixels
    points = {}
    for name, lm in landmarks.items():
        if lm['visibility'] > 0.5:
            x = int(lm['x'] * width)
            y = int(lm['y'] * height)
            points[name] = (x, y)
    
    # Draw connections
    for start_name, end_name in CONNECTIONS:
        if start_name in points and end_name in points:
            start_vis = landmarks[start_name]['visibility']
            end_vis = landmarks[end_name]['visibility']
            
            # Line thickness based on visibility
            thickness = 3 if min(start_vis, end_vis) > 0.7 else 2
            
            cv2.line(frame, points[start_name], points[end_name], color, thickness)
    
    # Draw joints
    for name, point in points.items():
        visibility = landmarks[name]['visibility']
        radius = 6 if visibility > 0.7 else 4
        cv2.circle(frame, point, radius, color, -1)
        cv2.circle(frame, point, radius + 2, (255, 255, 255), 1)


def draw_angles(frame, landmarks, metrics, width, height):
    """Draw angle measurements on joints"""
    
    # Left elbow angle
    if metrics['left_elbow_angle'] and 'left_elbow' in landmarks:
        x = int(landmarks['left_elbow']['x'] * width)
        y = int(landmarks['left_elbow']['y'] * height)
        angle_text = f"{metrics['left_elbow_angle']:.0f}"
        add_text(frame, angle_text, (x + 10, y - 10), COLORS['gold'], size=0.6)
    
    # Right elbow angle
    if metrics['right_elbow_angle'] and 'right_elbow' in landmarks:
        x = int(landmarks['right_elbow']['x'] * width)
        y = int(landmarks['right_elbow']['y'] * height)
        angle_text = f"{metrics['right_elbow_angle']:.0f}"
        add_text(frame, angle_text, (x + 10, y - 10), COLORS['gold'], size=0.6)


def add_text(frame, text, position, color, size=1.0, thickness=2):
    """Add text with background"""
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, size, thickness)
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(frame, 
                  (x - 5, y - text_height - 5), 
                  (x + text_width + 5, y + baseline + 5),
                  COLORS['background'], 
                  -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, size, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python module4_visualization.py <video> <pose_data.json> <quality_report.json> <metrics.json>")
        sys.exit(1)
    
    try:
        result = create_annotated_video(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        print("✓ SUCCESS!")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)