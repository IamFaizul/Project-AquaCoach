"""
MODULE 1: Pose Estimation
Detects pose landmarks and LEFT HAND stroke cycles

Stroke definition: Left hand completes full rotation and returns to starting position
"""

import cv2
import mediapipe as mp
import json
from pathlib import Path
import numpy as np


# Landmarks to track
LANDMARKS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
}


def extract_pose_landmarks(video_path, output_folder='output'):
    """
    Extract pose landmarks from video using MediaPipe
    
    Args:
        video_path: Path to video file
        output_folder: Where to save output JSON
    
    Returns:
        Path to output JSON file
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {Path(video_path).name}")
    print(f"{'='*60}\n")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Frames: {total_frames} | FPS: {fps}")
    print(f"{'='*60}\n")
    
    # Process frames
    frames_data = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(rgb_frame)
        
        # Extract landmarks
        landmarks = None
        if results.pose_landmarks:
            landmarks = {}
            for name, idx in LANDMARKS.items():
                lm = results.pose_landmarks.landmark[idx]
                landmarks[name] = {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                }
        
        frames_data.append({
            'frame_number': frame_count,
            'landmarks': landmarks
        })
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Progress: {frame_count}/{total_frames} frames...")
    
    cap.release()
    pose.close()
    
    # Apply smoothing
    print("\nApplying temporal smoothing...")
    apply_smoothing(frames_data)
    
    # Detect strokes (LEFT HAND ONLY)
    print("Detecting stroke cycles (LEFT HAND)...")
    strokes = detect_left_hand_strokes(frames_data, fps)
    
    # Calculate statistics
    stats = calculate_statistics(frames_data)
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Total frames: {len(frames_data)}")
    print(f"  Detected: {stats['detected_frames']} ({stats['detection_rate']*100:.1f}%)")
    print(f"  Null frames: {stats['null_frames']} ({stats['null_frame_ratio']*100:.1f}%)")
    print(f"  Low visibility: {stats['low_visibility_ratio']*100:.1f}%")
    print(f"  Discontinuities: {stats['discontinuities']}")
    print(f"  Left hand strokes found: {len(strokes)}")
    print(f"{'='*60}\n")
    
    # Save output
    Path(output_folder).mkdir(exist_ok=True)
    video_name = Path(video_path).stem
    
    output_data = {
        'video_info': {
            'source': video_path,
            'fps': fps,
            'total_frames': len(frames_data),
            'width': width,
            'height': height
        },
        'statistics': stats,
        'frames': frames_data,
        'cycles': strokes  # These are LEFT HAND strokes
    }
    
    output_path = Path(output_folder) / f"{video_name}_pose_data.json"
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ SAVED: {output_path}\n")
    
    return str(output_path)


def apply_smoothing(frames_data, window_size=3):
    """Apply temporal smoothing to reduce jitter"""
    
    for landmark_name in LANDMARKS.keys():
        # Collect positions
        positions = []
        for frame in frames_data:
            if frame['landmarks'] and landmark_name in frame['landmarks']:
                lm = frame['landmarks'][landmark_name]
                if lm['visibility'] >= 0.5:
                    positions.append({'x': lm['x'], 'y': lm['y'], 'z': lm['z']})
                else:
                    positions.append(None)
            else:
                positions.append(None)
        
        # Apply moving average
        half_window = window_size // 2
        
        for i in range(len(positions)):
            if positions[i] is None:
                continue
            
            # Collect window
            window = []
            for j in range(max(0, i - half_window), min(len(positions), i + half_window + 1)):
                if positions[j] is not None:
                    window.append(positions[j])
            
            if len(window) >= 2:
                # Average
                avg_x = np.mean([p['x'] for p in window])
                avg_y = np.mean([p['y'] for p in window])
                avg_z = np.mean([p['z'] for p in window])
                
                # Update
                if frames_data[i]['landmarks'] and landmark_name in frames_data[i]['landmarks']:
                    frames_data[i]['landmarks'][landmark_name]['x'] = avg_x
                    frames_data[i]['landmarks'][landmark_name]['y'] = avg_y
                    frames_data[i]['landmarks'][landmark_name]['z'] = avg_z


def detect_left_hand_strokes(frames_data, fps):
    """
    Detect strokes based on LEFT WRIST position
    
    One stroke = left wrist goes from high → low → high (full rotation)
    
    Args:
        frames_data: List of frame dicts
        fps: Frames per second
    
    Returns:
        List of stroke dicts
    """
    
    # Track left wrist Y position (vertical)
    y_positions = []
    
    for frame in frames_data:
        if (frame['landmarks'] and 
            'left_wrist' in frame['landmarks'] and
            frame['landmarks']['left_wrist']['visibility'] >= 0.4):
            y_positions.append(frame['landmarks']['left_wrist']['y'])
        else:
            y_positions.append(None)
    
    # Find peaks (highest points = hand at surface)
    peaks = []
    
    for i in range(1, len(y_positions) - 1):
        if y_positions[i] is None:
            continue
        
        # Get previous valid value
        prev_val = y_positions[i-1]
        if prev_val is None:
            for j in range(i-2, -1, -1):
                if y_positions[j] is not None:
                    prev_val = y_positions[j]
                    break
        
        # Get next valid value
        next_val = y_positions[i+1]
        if next_val is None:
            for j in range(i+2, len(y_positions)):
                if y_positions[j] is not None:
                    next_val = y_positions[j]
                    break
        
        # Check if peak (lowest Y = highest position in video coordinates)
        if prev_val is not None and next_val is not None:
            if y_positions[i] < prev_val and y_positions[i] < next_val:
                peaks.append(i)
    
    # Create strokes between peaks
    strokes = []
    min_stroke_length = int(0.5 * fps)  # Minimum 0.5 seconds per stroke
    
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        
        # Only count if long enough
        if end - start >= min_stroke_length:
            strokes.append({
                'cycle_id': len(strokes) + 1,
                'start_frame': start,
                'end_frame': end,
                'length_frames': end - start
            })
    
    return strokes


def calculate_statistics(frames_data):
    """Calculate quality statistics"""
    
    total_frames = len(frames_data)
    null_frames = sum(1 for f in frames_data if not f['landmarks'])
    detected_frames = total_frames - null_frames
    
    # Count low visibility
    low_vis_count = 0
    total_landmarks = 0
    
    for frame in frames_data:
        if frame['landmarks']:
            for lm in frame['landmarks'].values():
                total_landmarks += 1
                if lm['visibility'] < 0.5:
                    low_vis_count += 1
    
    # Count discontinuities
    discontinuities = 0
    prev_count = None
    
    for frame in frames_data:
        if frame['landmarks']:
            curr_count = len(frame['landmarks'])
            if prev_count is not None and curr_count != prev_count:
                discontinuities += 1
            prev_count = curr_count
    
    return {
        'total_frames': total_frames,
        'detected_frames': detected_frames,
        'null_frames': null_frames,
        'null_frame_ratio': null_frames / total_frames if total_frames > 0 else 0,
        'detection_rate': detected_frames / total_frames if total_frames > 0 else 0,
        'low_visibility_count': low_vis_count,
        'low_visibility_ratio': low_vis_count / total_landmarks if total_landmarks > 0 else 0,
        'discontinuities': discontinuities
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python module1_pose_estimation.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    try:
        result = extract_pose_landmarks(video_path)
        print(f"\n✓ SUCCESS! Output: {result}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)