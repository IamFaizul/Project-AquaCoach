"""
MODULE 3: ANGLE METRICS (OPTIMIZED)
Calculate swimming technique angles with high accuracy
NO STROKE COUNTING - ANGLES ONLY
"""

import json
import numpy as np
from pathlib import Path


def calculate_angle(point1, point2, point3):
    """
    Calculate angle at point2 formed by point1-point2-point3
    Returns angle in degrees
    """
    try:
        # Vectors
        v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
        v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
        
        # Angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    except:
        return None


def calculate_swimming_metrics(pose_data_path, quality_report_path, output_folder='output'):
    """Calculate swimming technique metrics"""
    
    print(f"\n{'='*60}")
    print(f"ANGLE METRICS CALCULATION")
    print(f"{'='*60}\n")
    
    # Load data
    with open(pose_data_path, 'r') as f:
        pose_data = json.load(f)
    
    with open(quality_report_path, 'r') as f:
        quality_data = json.load(f)
    
    video_name = Path(pose_data_path).stem.replace('_pose_data', '')
    frames = pose_data['frames']
    gold_frames = quality_data['validation']['gold_frame_indices']
    
    print(f"Video: {video_name}")
    print(f"Total frames: {len(frames)}")
    print(f"Gold frames: {len(gold_frames)}\n")
    
    # Calculate metrics for gold frames only
    print("Analyzing gold frames...")
    
    metrics = []
    
    for frame_idx in gold_frames:
        if frame_idx >= len(frames):
            continue
        
        frame = frames[frame_idx]
        
        if not frame['landmarks']:
            continue
        
        lm = frame['landmarks']
        
        # Initialize frame metrics
        frame_metrics = {
            'frame_number': frame_idx,
            'left_elbow_angle': None,
            'right_elbow_angle': None,
            'left_shoulder_angle': None,
            'right_shoulder_angle': None,
            'left_hip_angle': None,
            'right_hip_angle': None,
            'left_knee_angle': None,
            'right_knee_angle': None,
        }
        
        # LEFT ELBOW ANGLE (shoulder-elbow-wrist)
        if all(k in lm for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            if all(lm[k]['visibility'] > 0.7 for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                angle = calculate_angle(lm['left_shoulder'], lm['left_elbow'], lm['left_wrist'])
                if angle and 30 <= angle <= 180:  # Sanity check
                    frame_metrics['left_elbow_angle'] = round(angle, 2)
        
        # RIGHT ELBOW ANGLE
        if all(k in lm for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            if all(lm[k]['visibility'] > 0.7 for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                angle = calculate_angle(lm['right_shoulder'], lm['right_elbow'], lm['right_wrist'])
                if angle and 30 <= angle <= 180:
                    frame_metrics['right_elbow_angle'] = round(angle, 2)
        
        # LEFT SHOULDER ANGLE (elbow-shoulder-hip)
        if all(k in lm for k in ['left_elbow', 'left_shoulder', 'left_hip']):
            if all(lm[k]['visibility'] > 0.7 for k in ['left_elbow', 'left_shoulder', 'left_hip']):
                angle = calculate_angle(lm['left_elbow'], lm['left_shoulder'], lm['left_hip'])
                if angle and 30 <= angle <= 180:
                    frame_metrics['left_shoulder_angle'] = round(angle, 2)
        
        # RIGHT SHOULDER ANGLE
        if all(k in lm for k in ['right_elbow', 'right_shoulder', 'right_hip']):
            if all(lm[k]['visibility'] > 0.7 for k in ['right_elbow', 'right_shoulder', 'right_hip']):
                angle = calculate_angle(lm['right_elbow'], lm['right_shoulder'], lm['right_hip'])
                if angle and 30 <= angle <= 180:
                    frame_metrics['right_shoulder_angle'] = round(angle, 2)
        
        # LEFT HIP ANGLE (shoulder-hip-knee)
        if all(k in lm for k in ['left_shoulder', 'left_hip', 'left_knee']):
            if all(lm[k]['visibility'] > 0.7 for k in ['left_shoulder', 'left_hip', 'left_knee']):
                angle = calculate_angle(lm['left_shoulder'], lm['left_hip'], lm['left_knee'])
                if angle and 90 <= angle <= 180:
                    frame_metrics['left_hip_angle'] = round(angle, 2)
        
        # RIGHT HIP ANGLE
        if all(k in lm for k in ['right_shoulder', 'right_hip', 'right_knee']):
            if all(lm[k]['visibility'] > 0.7 for k in ['right_shoulder', 'right_hip', 'right_knee']):
                angle = calculate_angle(lm['right_shoulder'], lm['right_hip'], lm['right_knee'])
                if angle and 90 <= angle <= 180:
                    frame_metrics['right_hip_angle'] = round(angle, 2)
        
        # LEFT KNEE ANGLE (hip-knee-ankle)
        if all(k in lm for k in ['left_hip', 'left_knee']) and 'left_ankle' in lm:
            if all(lm[k]['visibility'] > 0.7 for k in ['left_hip', 'left_knee', 'left_ankle']):
                angle = calculate_angle(lm['left_hip'], lm['left_knee'], lm['left_ankle'])
                if angle and 90 <= angle <= 180:
                    frame_metrics['left_knee_angle'] = round(angle, 2)
        
        # RIGHT KNEE ANGLE
        if all(k in lm for k in ['right_hip', 'right_knee']) and 'right_ankle' in lm:
            if all(lm[k]['visibility'] > 0.7 for k in ['right_hip', 'right_knee', 'right_ankle']):
                angle = calculate_angle(lm['right_hip'], lm['right_knee'], lm['right_ankle'])
                if angle and 90 <= angle <= 180:
                    frame_metrics['right_knee_angle'] = round(angle, 2)
        
        metrics.append(frame_metrics)
    
    print(f"✓ Calculated metrics for {len(metrics)} frames\n")
    
    # Calculate aggregated statistics
    print("Calculating statistics...")
    
    stats = calculate_aggregate_stats(metrics)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    print("Average Angles (from gold frames):")
    print(f"  Left Elbow:     {stats['left_elbow']['mean']:.1f}° (±{stats['left_elbow']['std']:.1f}°)")
    print(f"  Right Elbow:    {stats['right_elbow']['mean']:.1f}° (±{stats['right_elbow']['std']:.1f}°)")
    print(f"  Left Shoulder:  {stats['left_shoulder']['mean']:.1f}° (±{stats['left_shoulder']['std']:.1f}°)")
    print(f"  Right Shoulder: {stats['right_shoulder']['mean']:.1f}° (±{stats['right_shoulder']['std']:.1f}°)")
    print(f"\nSymmetry:")
    print(f"  Elbow difference:    {abs(stats['left_elbow']['mean'] - stats['right_elbow']['mean']):.1f}°")
    print(f"  Shoulder difference: {abs(stats['left_shoulder']['mean'] - stats['right_shoulder']['mean']):.1f}°")
    
    # Save output
    output_data = {
        'video_name': video_name,
        'total_gold_frames': len(gold_frames),
        'frames_with_metrics': len(metrics),
        'frame_metrics': metrics,
        'aggregate_statistics': stats
    }
    
    Path(output_folder).mkdir(exist_ok=True)
    output_path = Path(output_folder) / f"{video_name}_metrics.json"
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ SAVED: {output_path}\n")
    
    return str(output_path)


def calculate_aggregate_stats(metrics):
    """Calculate mean, std, min, max for all angles"""
    
    angle_types = [
        'left_elbow_angle', 'right_elbow_angle',
        'left_shoulder_angle', 'right_shoulder_angle',
        'left_hip_angle', 'right_hip_angle',
        'left_knee_angle', 'right_knee_angle'
    ]
    
    stats = {}
    
    for angle_type in angle_types:
        values = [m[angle_type] for m in metrics if m[angle_type] is not None]
        
        if values:
            stats[angle_type.replace('_angle', '')] = {
                'mean': round(np.mean(values), 2),
                'std': round(np.std(values), 2),
                'min': round(np.min(values), 2),
                'max': round(np.max(values), 2),
                'count': len(values)
            }
        else:
            stats[angle_type.replace('_angle', '')] = {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'count': 0
            }
    
    return stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python module3_angle_metrics.py <pose_data.json> <quality_report.json>")
        sys.exit(1)
    
    try:
        result = calculate_swimming_metrics(sys.argv[1], sys.argv[2])
        print("✓ SUCCESS!")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)