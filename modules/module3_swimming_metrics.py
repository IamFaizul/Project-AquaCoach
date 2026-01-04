"""
MODULE 3: Swimming Metrics
Calculates swimming-specific measurements from validated gold frames

Input: 
  - pose_data.json (from Module 1)
  - quality_report.json (from Module 2)
Output:
  - swimming_metrics.json
"""

import json
from pathlib import Path
import numpy as np


# ==================== HELPER FUNCTIONS ====================

def calculate_angle(point1, point2, point3):
    """
    Calculate angle at point2 formed by point1-point2-point3
    Returns angle in degrees (0-180)
    """
    # Vectors
    v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
    v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
    
    # Angle calculation
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(
        (point1['x'] - point2['x'])**2 + 
        (point1['y'] - point2['y'])**2
    )


# ==================== METRIC CALCULATIONS ====================

def calculate_elbow_angle(landmarks, side='left'):
    """
    Calculate elbow angle for left or right arm
    
    Returns: angle in degrees or None if landmarks missing
    """
    shoulder_key = f'{side}_shoulder'
    elbow_key = f'{side}_elbow'
    wrist_key = f'{side}_wrist'
    
    # Check all landmarks present
    if not all(k in landmarks for k in [shoulder_key, elbow_key, wrist_key]):
        return None
    
    shoulder = landmarks[shoulder_key]
    elbow = landmarks[elbow_key]
    wrist = landmarks[wrist_key]
    
    # Check visibility
    if any(lm['visibility'] < 0.6 for lm in [shoulder, elbow, wrist]):
        return None
    
    angle = calculate_angle(shoulder, elbow, wrist)
    return angle


def calculate_hip_angle(landmarks, side='left'):
    """
    Calculate hip angle for left or right side
    
    Returns: angle in degrees or None if landmarks missing
    """
    shoulder_key = f'{side}_shoulder'
    hip_key = f'{side}_hip'
    knee_key = f'{side}_knee'
    
    # Check all landmarks present
    if not all(k in landmarks for k in [shoulder_key, hip_key, knee_key]):
        return None
    
    shoulder = landmarks[shoulder_key]
    hip = landmarks[hip_key]
    knee = landmarks[knee_key]
    
    # Check visibility
    if any(lm['visibility'] < 0.6 for lm in [shoulder, hip, knee]):
        return None
    
    angle = calculate_angle(shoulder, hip, knee)
    return angle


def calculate_body_alignment(landmarks):
    """
    Calculate how horizontal the body is (swimming position)
    
    Returns: tilt angle in degrees (0 = perfectly horizontal)
    """
    if not all(k in landmarks for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
        return None
    
    shoulder_y = (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2
    hip_y = (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2
    
    tilt = abs(shoulder_y - hip_y)
    return tilt * 100  # Convert to percentage


def calculate_stroke_symmetry(left_angle, right_angle):
    """
    Calculate symmetry between left and right measurements
    
    Returns: difference in degrees (0 = perfect symmetry)
    """
    if left_angle is None or right_angle is None:
        return None
    
    return abs(left_angle - right_angle)


# ==================== ANALYZE GOLD FRAMES ====================

def analyze_gold_frames(pose_data, gold_frame_numbers):
    """
    Analyze metrics from gold frames only
    
    Returns: Dict with all calculated metrics
    """
    
    frames = pose_data['frames']
    fps = pose_data['video_info']['fps']
    
    # Collect metrics from each gold frame
    metrics_per_frame = []
    
    print(f"\nAnalyzing {len(gold_frame_numbers)} gold frames...")
    
    for frame_num in gold_frame_numbers:
        if frame_num >= len(frames):
            continue
        
        frame = frames[frame_num]
        landmarks = frame.get('landmarks')
        
        if not landmarks:
            continue
        
        # Calculate all metrics for this frame
        frame_metrics = {
            'frame_number': frame_num,
            'left_elbow': calculate_elbow_angle(landmarks, 'left'),
            'right_elbow': calculate_elbow_angle(landmarks, 'right'),
            'left_hip': calculate_hip_angle(landmarks, 'left'),
            'right_hip': calculate_hip_angle(landmarks, 'right'),
            'body_alignment': calculate_body_alignment(landmarks)
        }
        
        # Calculate symmetry
        if frame_metrics['left_elbow'] and frame_metrics['right_elbow']:
            frame_metrics['elbow_symmetry'] = calculate_stroke_symmetry(
                frame_metrics['left_elbow'],
                frame_metrics['right_elbow']
            )
        else:
            frame_metrics['elbow_symmetry'] = None
        
        metrics_per_frame.append(frame_metrics)
    
    print(f"✓ Calculated metrics for {len(metrics_per_frame)} frames")
    
    # Aggregate statistics
    return aggregate_metrics(metrics_per_frame, fps)


def aggregate_metrics(metrics_per_frame, fps):
    """
    Calculate aggregate statistics from per-frame metrics
    
    Returns: Summary statistics
    """
    
    # Helper to calculate stats for a metric
    def calc_stats(metric_name):
        values = [m[metric_name] for m in metrics_per_frame if m[metric_name] is not None]
        
        if not values:
            return None
        
        return {
            'mean': round(float(np.mean(values)), 2),
            'std': round(float(np.std(values)), 2),
            'min': round(float(np.min(values)), 2),
            'max': round(float(np.max(values)), 2),
            'median': round(float(np.median(values)), 2),
            'count': len(values)
        }
    
    # Calculate stats for each metric
    aggregated = {
        'left_elbow_angle': calc_stats('left_elbow'),
        'right_elbow_angle': calc_stats('right_elbow'),
        'left_hip_angle': calc_stats('left_hip'),
        'right_hip_angle': calc_stats('right_hip'),
        'body_alignment': calc_stats('body_alignment'),
        'elbow_symmetry': calc_stats('elbow_symmetry')
    }
    
    return aggregated


# ==================== ANALYZE CYCLES ====================

def analyze_cycles(pose_data, gold_frame_numbers):
    """
    Analyze each detected stroke cycle
    
    Returns: List of cycle metrics
    """
    
    cycles = pose_data.get('cycles', [])
    frames = pose_data['frames']
    fps = pose_data['video_info']['fps']
    
    if not cycles:
        return []
    
    print(f"\nAnalyzing {len(cycles)} stroke cycles...")
    
    cycle_metrics = []
    
    for cycle in cycles:
        cycle_id = cycle['cycle_id']
        start_frame = cycle['start_frame']
        end_frame = cycle['end_frame']
        
        # Get frames in this cycle that are ALSO gold frames
        cycle_gold_frames = [
            f for f in range(start_frame, end_frame + 1)
            if f in gold_frame_numbers
        ]
        
        if len(cycle_gold_frames) < 10:  # Need minimum frames for reliable analysis
            continue
        
        # Collect metrics for this cycle
        cycle_frame_metrics = []
        
        for frame_num in cycle_gold_frames:
            if frame_num >= len(frames):
                continue
            
            frame = frames[frame_num]
            landmarks = frame.get('landmarks')
            
            if not landmarks:
                continue
            
            metrics = {
                'left_elbow': calculate_elbow_angle(landmarks, 'left'),
                'right_elbow': calculate_elbow_angle(landmarks, 'right'),
                'left_hip': calculate_hip_angle(landmarks, 'left'),
                'right_hip': calculate_hip_angle(landmarks, 'right'),
            }
            
            cycle_frame_metrics.append(metrics)
        
        if not cycle_frame_metrics:
            continue
        
        # Calculate average for this cycle
        def avg_metric(metric_name):
            values = [m[metric_name] for m in cycle_frame_metrics if m[metric_name] is not None]
            return round(float(np.mean(values)), 2) if values else None
        
        cycle_summary = {
            'cycle_id': cycle_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'duration_sec': round((end_frame - start_frame) / fps, 2),
            'gold_frames_in_cycle': len(cycle_gold_frames),
            'left_elbow_avg': avg_metric('left_elbow'),
            'right_elbow_avg': avg_metric('right_elbow'),
            'left_hip_avg': avg_metric('left_hip'),
            'right_hip_avg': avg_metric('right_hip')
        }
        
        # Calculate symmetry for this cycle
        if cycle_summary['left_elbow_avg'] and cycle_summary['right_elbow_avg']:
            cycle_summary['elbow_symmetry'] = round(
                abs(cycle_summary['left_elbow_avg'] - cycle_summary['right_elbow_avg']), 
                2
            )
        
        cycle_metrics.append(cycle_summary)
    
    print(f"✓ Analyzed {len(cycle_metrics)} cycles with sufficient gold frames")
    
    return cycle_metrics


def find_best_cycle(cycle_metrics):
    """
    Find the best quality stroke cycle
    
    Best = most gold frames + best symmetry
    """
    if not cycle_metrics:
        return None
    
    # Score each cycle
    scored_cycles = []
    
    for cycle in cycle_metrics:
        score = 0
        
        # More gold frames = better
        score += cycle['gold_frames_in_cycle']
        
        # Better symmetry = better (lower is better, so subtract)
        if cycle.get('elbow_symmetry') is not None:
            score -= cycle['elbow_symmetry'] / 10  # Small penalty for asymmetry
        
        scored_cycles.append((score, cycle))
    
    # Return cycle with highest score
    best = max(scored_cycles, key=lambda x: x[0])
    return best[1]


# ==================== STROKE COUNT (FIXED) ====================

def calculate_stroke_metrics(cycles, video_duration_sec):
    """
    Calculate stroke count
    
    FIXED: One cycle = one stroke (one arm completes full rotation)
    NOT cycles * 2
    
    Args:
        cycles: List of analyzed cycles
        video_duration_sec: Total video duration
    
    Returns: Dict with stroke count only
    """
    
    if not cycles:
        return {'stroke_count': 0}
    
    # FIXED: One cycle = one stroke
    # Do NOT multiply by 2
    stroke_count = len(cycles)
    
    return {'stroke_count': stroke_count}


# ==================== MAIN FUNCTION ====================

def calculate_swimming_metrics(pose_data_path, quality_report_path, output_folder='output'):
    """
    Main function: Calculate swimming metrics from validated data
    
    Args:
        pose_data_path: Path to pose_data.json
        quality_report_path: Path to quality_report.json
        output_folder: Where to save results
    
    Returns:
        Path to swimming metrics JSON
    """
    
    # Load data
    print(f"\n{'='*60}")
    print(f"LOADING DATA")
    print(f"{'='*60}\n")
    
    with open(pose_data_path, 'r') as f:
        pose_data = json.load(f)
    
    with open(quality_report_path, 'r') as f:
        quality_report = json.load(f)
    
    video_name = quality_report['video_name']
    gold_frames = quality_report['validation_results']['gold_frames']
    video_duration = len(pose_data['frames']) / pose_data['video_info']['fps']
    
    print(f"Video: {video_name}")
    print(f"Total frames: {len(pose_data['frames'])}")
    print(f"Gold frames: {len(gold_frames)} ({len(gold_frames)/len(pose_data['frames'])*100:.1f}%)")
    print(f"Duration: {video_duration:.1f} seconds")
    
    # Analyze gold frames
    print(f"\n{'='*60}")
    print(f"ANALYZING GOLD FRAMES")
    print(f"{'='*60}")
    
    overall_metrics = analyze_gold_frames(pose_data, gold_frames)
    
    # Analyze cycles
    print(f"\n{'='*60}")
    print(f"ANALYZING STROKE CYCLES")
    print(f"{'='*60}")
    
    cycle_metrics = analyze_cycles(pose_data, gold_frames)
    best_cycle = find_best_cycle(cycle_metrics)
    
    # Calculate stroke count (FIXED)
    print(f"\n{'='*60}")
    print(f"CALCULATING STROKE COUNT")
    print(f"{'='*60}")
    
    stroke_metrics = calculate_stroke_metrics(cycle_metrics, video_duration)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    # Stroke information
    print(f"Stroke Information:")
    print(f"  Total strokes: {stroke_metrics['stroke_count']}")
    
    # Overall metrics
    if overall_metrics.get('left_elbow_angle'):
        print(f"\nOverall Metrics (from {len(gold_frames)} gold frames):")
        print(f"  Left Elbow:  {overall_metrics['left_elbow_angle']['mean']}° (±{overall_metrics['left_elbow_angle']['std']}°)")
    
    if overall_metrics.get('right_elbow_angle'):
        print(f"  Right Elbow: {overall_metrics['right_elbow_angle']['mean']}° (±{overall_metrics['right_elbow_angle']['std']}°)")
    
    if overall_metrics.get('elbow_symmetry'):
        print(f"  Symmetry:    {overall_metrics['elbow_symmetry']['mean']}° difference")
    
    # Best cycle
    if best_cycle:
        print(f"\nBest Cycle (Cycle #{best_cycle['cycle_id']}):")
        print(f"  Duration: {best_cycle['duration_sec']}s")
        print(f"  Left Elbow: {best_cycle['left_elbow_avg']}°")
        print(f"  Right Elbow: {best_cycle['right_elbow_avg']}°")
        print(f"  Symmetry: {best_cycle.get('elbow_symmetry', 'N/A')}°")
    
    print(f"\n{'='*60}\n")
    
    # Save results
    Path(output_folder).mkdir(exist_ok=True)
    
    output_data = {
        'video_name': video_name,
        'stroke_metrics': stroke_metrics,
        'overall_metrics': overall_metrics,
        'cycle_analysis': {
            'total_cycles': len(cycle_metrics),
            'cycles': cycle_metrics,
            'best_cycle': best_cycle
        },
        'metadata': {
            'gold_frames_analyzed': len(gold_frames),
            'total_frames': len(pose_data['frames']),
            'video_duration_sec': round(video_duration, 2),
            'quality_score': quality_report['overall_quality_score']
        }
    }
    
    output_path = Path(output_folder) / f"{video_name}_swimming_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ SAVED: {output_path}\n")
    
    return str(output_path)


# ==================== CLI ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python module3_swimming_metrics.py <pose_data.json> <quality_report.json>")
        sys.exit(1)
    
    pose_data_path = sys.argv[1]
    quality_report_path = sys.argv[2]
    
    try:
        result = calculate_swimming_metrics(pose_data_path, quality_report_path)
        print(f"\n✓ SUCCESS! Swimming metrics: {result}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()