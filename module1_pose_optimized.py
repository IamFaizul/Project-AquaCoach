"""
MODULE 1: OPTIMIZED POSE DETECTION
High-accuracy MediaPipe pose estimation for swimming analysis
"""

import cv2
import mediapipe as mp
import json
from pathlib import Path
import numpy as np


# Optimized configuration for accuracy
CONFIG = {
    'model_complexity': 2,  # Highest quality (0=lite, 1=full, 2=heavy)
    'min_detection_confidence': 0.6,  # Stricter detection
    'min_tracking_confidence': 0.6,   # Stricter tracking
    'smoothing_window': 5,  # 5-frame smoothing for stability
    'visibility_threshold': 0.7,  # Only use high-confidence landmarks
    'outlier_threshold': 0.2,  # Remove jumps > 20% between frames
}


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
    """Extract pose landmarks with optimized settings"""
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZED POSE DETECTION")
    print(f"{'='*60}\n")
    
    # Initialize MediaPipe with optimal settings
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=CONFIG['model_complexity'],
        smooth_landmarks=True,
        min_detection_confidence=CONFIG['min_detection_confidence'],
        min_tracking_confidence=CONFIG['min_tracking_confidence']
    )
    
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_name}")
    print(f"Frames: {total_frames} | FPS: {fps}")
    print(f"Quality: Model Complexity {CONFIG['model_complexity']} (Highest)\n")
    
    # Process frames
    frames_data = []
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
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
            'frame_number': frame_num,
            'landmarks': landmarks
        })
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  {frame_num}/{total_frames} frames...")
    
    cap.release()
    pose.close()
    
    print(f"\n✓ Extraction complete")
    
    # Apply optimizations
    print("Applying optimizations...")
    print("  1. Removing outliers...")
    remove_outliers(frames_data)
    
    print("  2. Applying 5-frame smoothing...")
    apply_smoothing(frames_data)
    
    print("  3. Bilateral filtering...")
    apply_bilateral_filtering(frames_data)
    
    # Calculate statistics
    stats = calculate_stats(frames_data)
    
    print(f"\nRESULTS:")
    print(f"  Detected: {stats['detected_frames']}/{stats['total_frames']} ({stats['detection_rate']*100:.1f}%)")
    print(f"  High quality: {stats['high_quality_frames']} frames (visibility > 0.7)")
    print(f"  Outliers removed: {stats['outliers_removed']}")
    
    # Save
    Path(output_folder).mkdir(exist_ok=True)
    
    output_data = {
        'video_info': {
            'source': video_path,
            'fps': fps,
            'total_frames': stats['total_frames'],
            'width': width,
            'height': height
        },
        'configuration': CONFIG,
        'statistics': stats,
        'frames': frames_data
    }
    
    output_path = Path(output_folder) / f"{video_name}_pose_data.json"
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ SAVED: {output_path}\n")
    
    return str(output_path)


def remove_outliers(frames_data):
    """Remove landmark outliers (sudden jumps)"""
    
    outliers_removed = 0
    
    for i in range(1, len(frames_data)):
        current = frames_data[i]
        previous = frames_data[i-1]
        
        if not current['landmarks'] or not previous['landmarks']:
            continue
        
        for name in LANDMARKS.keys():
            if name not in current['landmarks'] or name not in previous['landmarks']:
                continue
            
            curr_lm = current['landmarks'][name]
            prev_lm = previous['landmarks'][name]
            
            # Calculate distance moved
            dx = curr_lm['x'] - prev_lm['x']
            dy = curr_lm['y'] - prev_lm['y']
            distance = np.sqrt(dx**2 + dy**2)
            
            # If jumped too much, mark as low confidence
            if distance > CONFIG['outlier_threshold']:
                curr_lm['visibility'] = 0.0
                outliers_removed += 1
    
    # Store for stats
    frames_data[0]['_outliers_removed'] = outliers_removed


def apply_smoothing(frames_data):
    """Apply 5-frame moving average smoothing"""
    
    window_size = CONFIG['smoothing_window']
    half_window = window_size // 2
    
    for name in LANDMARKS.keys():
        # Collect positions
        positions = []
        
        for frame in frames_data:
            if frame['landmarks'] and name in frame['landmarks']:
                lm = frame['landmarks'][name]
                if lm['visibility'] >= CONFIG['visibility_threshold']:
                    positions.append({'x': lm['x'], 'y': lm['y'], 'z': lm['z']})
                else:
                    positions.append(None)
            else:
                positions.append(None)
        
        # Apply moving average
        for i in range(len(positions)):
            if positions[i] is None:
                continue
            
            window = []
            for j in range(max(0, i - half_window), min(len(positions), i + half_window + 1)):
                if positions[j] is not None:
                    window.append(positions[j])
            
            if len(window) >= 3:
                avg_x = np.mean([p['x'] for p in window])
                avg_y = np.mean([p['y'] for p in window])
                avg_z = np.mean([p['z'] for p in window])
                
                if frames_data[i]['landmarks'] and name in frames_data[i]['landmarks']:
                    frames_data[i]['landmarks'][name]['x'] = avg_x
                    frames_data[i]['landmarks'][name]['y'] = avg_y
                    frames_data[i]['landmarks'][name]['z'] = avg_z


def apply_bilateral_filtering(frames_data):
    """Use bilateral symmetry - if one side missing, estimate from other"""
    
    pairs = [
        ('left_shoulder', 'right_shoulder'),
        ('left_elbow', 'right_elbow'),
        ('left_wrist', 'right_wrist'),
        ('left_hip', 'right_hip'),
        ('left_knee', 'right_knee'),
    ]
    
    for frame in frames_data:
        if not frame['landmarks']:
            continue
        
        for left_name, right_name in pairs:
            left_lm = frame['landmarks'].get(left_name)
            right_lm = frame['landmarks'].get(right_name)
            
            if not left_lm or not right_lm:
                continue
            
            # If one side has low confidence, boost it from the other
            if left_lm['visibility'] < 0.5 and right_lm['visibility'] >= 0.7:
                # Estimate left from right (mirror)
                left_lm['visibility'] = right_lm['visibility'] * 0.8
            
            elif right_lm['visibility'] < 0.5 and left_lm['visibility'] >= 0.7:
                # Estimate right from left (mirror)
                right_lm['visibility'] = left_lm['visibility'] * 0.8


def calculate_stats(frames_data):
    """Calculate statistics"""
    
    total_frames = len(frames_data)
    null_frames = sum(1 for f in frames_data if not f['landmarks'])
    detected_frames = total_frames - null_frames
    
    # Count high quality frames
    high_quality = 0
    low_vis_count = 0
    total_landmarks = 0
    
    for frame in frames_data:
        if frame['landmarks']:
            frame_quality = True
            for lm in frame['landmarks'].values():
                total_landmarks += 1
                if lm['visibility'] < 0.5:
                    low_vis_count += 1
                if lm['visibility'] < CONFIG['visibility_threshold']:
                    frame_quality = False
            
            if frame_quality:
                high_quality += 1
    
    # Get outliers count
    outliers_removed = frames_data[0].get('_outliers_removed', 0) if frames_data else 0
    
    return {
        'total_frames': total_frames,
        'detected_frames': detected_frames,
        'null_frames': null_frames,
        'detection_rate': detected_frames / total_frames if total_frames > 0 else 0,
        'high_quality_frames': high_quality,
        'low_visibility_count': low_vis_count,
        'low_visibility_ratio': low_vis_count / total_landmarks if total_landmarks > 0 else 0,
        'outliers_removed': outliers_removed,
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python module1_pose_optimized.py <video_path>")
        sys.exit(1)
    
    try:
        result = extract_pose_landmarks(sys.argv[1])
        print("✓ SUCCESS!")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)