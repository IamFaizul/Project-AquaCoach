"""
MODULE 3.5: Visualization (Simplified)
Draw skeleton + angles + stroke count on video

Input: 
  - Video file
  - pose_data.json
  - quality_report.json
  - swimming_metrics.json
Output:
  - Annotated video with skeleton overlay + stroke count
"""

import cv2
import json
from pathlib import Path
import numpy as np


# Colors (BGR format for OpenCV)
COLORS = {
    'landmark': (0, 255, 0),      # Green
    'bone': (255, 200, 0),         # Cyan
    'gold_frame': (0, 255, 0),     # Green
    'normal_frame': (0, 165, 255), # Orange
    'text': (0, 0, 255),           # Red
    'background': (0, 0, 0),       # Black
    'white': (255, 255, 255)       # White
}

# Skeleton connections
SKELETON = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
]


def calculate_angle(point1, point2, point3):
    """Calculate angle at point2"""
    v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
    v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle


def draw_skeleton(frame, landmarks, is_gold_frame):
    """Draw skeleton overlay on frame"""
    height, width = frame.shape[:2]
    
    bone_color = COLORS['gold_frame'] if is_gold_frame else COLORS['normal_frame']
    
    # Draw bones
    for joint1, joint2 in SKELETON:
        if joint1 in landmarks and joint2 in landmarks:
            lm1 = landmarks[joint1]
            lm2 = landmarks[joint2]
            
            if lm1['visibility'] >= 0.6 and lm2['visibility'] >= 0.6:
                pt1 = (int(lm1['x'] * width), int(lm1['y'] * height))
                pt2 = (int(lm2['x'] * width), int(lm2['y'] * height))
                cv2.line(frame, pt1, pt2, bone_color, 2)
    
    # Draw landmarks
    for name, lm in landmarks.items():
        if lm['visibility'] >= 0.6:
            pt = (int(lm['x'] * width), int(lm['y'] * height))
            cv2.circle(frame, pt, 5, COLORS['landmark'], -1)
            cv2.circle(frame, pt, 6, (0, 0, 0), 1)
    
    return frame


def draw_angles(frame, landmarks):
    """Calculate and draw angle values on frame"""
    height, width = frame.shape[:2]
    angles_to_draw = []
    
    # Left elbow
    if all(k in landmarks for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
        if all(landmarks[k]['visibility'] >= 0.6 for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            angle = calculate_angle(
                landmarks['left_shoulder'],
                landmarks['left_elbow'],
                landmarks['left_wrist']
            )
            elbow_pos = landmarks['left_elbow']
            text_pos = (int(elbow_pos['x'] * width) - 30, int(elbow_pos['y'] * height) - 10)
            angles_to_draw.append((f"L: {int(angle)}°", text_pos))
    
    # Right elbow
    if all(k in landmarks for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        if all(landmarks[k]['visibility'] >= 0.6 for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            angle = calculate_angle(
                landmarks['right_shoulder'],
                landmarks['right_elbow'],
                landmarks['right_wrist']
            )
            elbow_pos = landmarks['right_elbow']
            text_pos = (int(elbow_pos['x'] * width) + 10, int(elbow_pos['y'] * height) - 10)
            angles_to_draw.append((f"R: {int(angle)}°", text_pos))
    
    # Left hip
    if all(k in landmarks for k in ['left_shoulder', 'left_hip', 'left_knee']):
        if all(landmarks[k]['visibility'] >= 0.6 for k in ['left_shoulder', 'left_hip', 'left_knee']):
            angle = calculate_angle(
                landmarks['left_shoulder'],
                landmarks['left_hip'],
                landmarks['left_knee']
            )
            hip_pos = landmarks['left_hip']
            text_pos = (int(hip_pos['x'] * width) - 40, int(hip_pos['y'] * height))
            angles_to_draw.append((f"Hip: {int(angle)}°", text_pos))
    
    # Draw all angle texts
    for text, pos in angles_to_draw:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            frame,
            (pos[0] - 2, pos[1] - text_size[1] - 2),
            (pos[0] + text_size[0] + 2, pos[1] + 2),
            COLORS['background'],
            -1
        )
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 2)
    
    return frame


def draw_info_panel(frame, frame_num, is_gold, cycle_id, stroke_count):
    """
    Draw information panel on frame (SIMPLIFIED)
    """
    height, width = frame.shape[:2]
    
    # Main info panel
    cv2.rectangle(frame, (10, 10), (320, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (320, 120), (255, 255, 255), 2)
    
    y_offset = 35
    
    # Frame number
    cv2.putText(frame, f"Frame: {frame_num}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    
    # Gold frame status
    status_color = (0, 255, 0) if is_gold else (0, 165, 255)
    status_text = "GOLD FRAME" if is_gold else "EXCLUDED"
    cv2.putText(frame, status_text, (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    y_offset += 25
    
    # Cycle info
    if cycle_id is not None:
        cv2.putText(frame, f"Stroke Cycle: #{cycle_id}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        y_offset += 25
    
    # Stroke count (SIMPLIFIED - ONLY STROKE COUNT)
    if stroke_count is not None:
        cv2.putText(frame, f"Total Strokes: {stroke_count}", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['white'], 2)
    
    return frame


def create_annotated_video(video_path, pose_data_path, quality_report_path, 
                          metrics_path=None, output_folder='output'):
    """Create annotated video with skeleton overlay + stroke count"""
    
    # Load data
    print(f"\n{'='*60}")
    print(f"LOADING DATA")
    print(f"{'='*60}\n")
    
    with open(pose_data_path, 'r') as f:
        pose_data = json.load(f)
    
    with open(quality_report_path, 'r') as f:
        quality_report = json.load(f)
    
    # Load swimming metrics if available
    stroke_count = None
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path, 'r') as f:
            swimming_metrics = json.load(f)
            stroke_count = swimming_metrics.get('stroke_metrics', {}).get('stroke_count')
    
    video_name = quality_report['video_name']
    gold_frames = set(quality_report['validation_results']['gold_frames'])
    frames_data = pose_data['frames']
    cycles = pose_data.get('cycles', [])
    
    print(f"Video: {video_name}")
    print(f"Gold frames: {len(gold_frames)}")
    if stroke_count:
        print(f"Total strokes: {stroke_count}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video
    Path(output_folder).mkdir(exist_ok=True)
    output_path = Path(output_folder) / f"{video_name}_annotated.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"\n{'='*60}")
    print(f"CREATING ANNOTATED VIDEO")
    print(f"{'='*60}\n")
    
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get landmarks for this frame
        if frame_num < len(frames_data):
            frame_data = frames_data[frame_num]
            landmarks = frame_data.get('landmarks')
            is_gold = frame_num in gold_frames
            
            # Find which cycle this frame belongs to
            cycle_id = None
            for cycle in cycles:
                if cycle['start_frame'] <= frame_num <= cycle['end_frame']:
                    cycle_id = cycle['cycle_id']
                    break
            
            # Draw annotations
            if landmarks:
                frame = draw_skeleton(frame, landmarks, is_gold)
                frame = draw_angles(frame, landmarks)
            
            frame = draw_info_panel(frame, frame_num, is_gold, cycle_id, stroke_count)
        
        # Write frame
        out.write(frame)
        
        frame_num += 1
        if frame_num % 50 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"\n{'='*60}")
    print(f"✓ SAVED: {output_path}")
    print(f"{'='*60}\n")
    
    return str(output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python module3_visualize.py <video> <pose_data.json> <quality_report.json> [swimming_metrics.json]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    pose_data_path = sys.argv[2]
    quality_report_path = sys.argv[3]
    metrics_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    try:
        result = create_annotated_video(video_path, pose_data_path, quality_report_path, metrics_path)
        print(f"\n✓ SUCCESS! Annotated video: {result}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()