"""
MODULE 2: QUALITY SCORING
Validates pose data quality and scores video usability
"""

import json
import numpy as np
from pathlib import Path


def validate_and_score(pose_data_path, output_folder='output'):
    """
    Validate pose data and calculate quality score
    
    Args:
        pose_data_path: Path to pose data JSON
        output_folder: Output directory
    
    Returns:
        Path to quality report JSON
    """
    
    print(f"\n{'='*60}")
    print(f"QUALITY SCORING")
    print(f"{'='*60}\n")
    
    # Load pose data
    with open(pose_data_path, 'r') as f:
        data = json.load(f)
    
    video_name = Path(pose_data_path).stem.replace('_pose_data', '')
    frames = data['frames']
    
    print(f"Video: {video_name}")
    print(f"Total frames: {len(frames)}\n")
    
    # Validation (2-layer approach)
    print("Validating frames (2-layer approach)...")
    validation_results = validate_frames(frames)
    
    print(f"✓ Validation complete\n")
    
    # Quality scoring
    print("Calculating quality score...")
    quality_score = calculate_quality_score(frames, validation_results)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Frames with good visibility: {validation_results['good_visibility_count']} ({validation_results['good_visibility_count']/len(frames)*100:.1f}%)")
    print(f"\n  🏆 GOLD FRAMES: {validation_results['gold_frame_count']} ({validation_results['gold_frame_ratio']*100:.1f}%)")
    print(f"{'='*60}\n")
    
    print(f"{'='*60}")
    print(f"QUALITY SCORING")
    print(f"{'='*60}\n")
    
    print(f"Scoring Breakdown:")
    print(f"  Landmark Fidelity: {quality_score['landmark_fidelity']:.1f}/25")
    print(f"  Kinematic Coherence: {quality_score['kinematic_coherence']:.1f}/25")
    print(f"  Context Discrimination: {quality_score['context_discrimination']:.1f}/25")
    print(f"  Hallucination Risk: {quality_score['hallucination_risk']:.1f}/25")
    print(f"\n  📊 OVERALL SCORE: {quality_score['overall_score']:.1f}/100")
    
    # Determine status
    if quality_score['overall_score'] >= 70:
        status = "USABLE"
        emoji = "✅"
    elif quality_score['overall_score'] >= 50:
        status = "NEEDS IMPROVEMENT"
        emoji = "⚠️"
    else:
        status = "POOR QUALITY"
        emoji = "❌"
    
    print(f"\n  {emoji} Video Status: {status}")
    print(f"{'='*60}\n")
    
    # Save output
    output_data = {
        'video_name': video_name,
        'validation': validation_results,
        'quality_score': quality_score,
        'status': status
    }
    
    Path(output_folder).mkdir(exist_ok=True)
    output_path = Path(output_folder) / f"{video_name}_quality_report.json"
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ SAVED: {output_path}\n")
    
    return str(output_path)


def validate_frames(frames):
    """
    2-Layer Validation Approach
    
    Layer 1: Good Visibility (visibility > 0.5 for all landmarks)
    Layer 2: Gold Standard (visibility > 0.7 for all landmarks)
    """
    
    good_visibility_indices = []
    gold_frame_indices = []
    
    for i, frame in enumerate(frames):
        if not frame['landmarks']:
            continue
        
        # Check visibility for all landmarks
        visibilities = [lm['visibility'] for lm in frame['landmarks'].values()]
        
        # Layer 1: Good visibility (all > 0.5)
        if all(v > 0.5 for v in visibilities):
            good_visibility_indices.append(i)
            
            # Layer 2: Gold standard (all > 0.7)
            if all(v > 0.7 for v in visibilities):
                gold_frame_indices.append(i)
    
    return {
        'good_visibility_indices': good_visibility_indices,
        'good_visibility_count': len(good_visibility_indices),
        'gold_frame_indices': gold_frame_indices,
        'gold_frame_count': len(gold_frame_indices),
        'gold_frame_ratio': len(gold_frame_indices) / len(frames) if len(frames) > 0 else 0
    }


def calculate_quality_score(frames, validation_results):
    """
    Calculate quality score (0-100)
    
    Four components (each 25 points):
    1. Landmark Fidelity (detection rate)
    2. Kinematic Coherence (temporal consistency)
    3. Context Discrimination (gold frame ratio)
    4. Hallucination Risk (inverse of low visibility)
    """
    
    total_frames = len(frames)
    detected_frames = sum(1 for f in frames if f['landmarks'])
    
    # 1. Landmark Fidelity (25 points)
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0
    landmark_fidelity = detection_rate * 25
    
    # 2. Kinematic Coherence (25 points)
    # Measure temporal consistency (fewer discontinuities = better)
    discontinuities = 0
    prev_count = None
    
    for frame in frames:
        if frame['landmarks']:
            curr_count = len(frame['landmarks'])
            if prev_count is not None and curr_count != prev_count:
                discontinuities += 1
            prev_count = curr_count
    
    discontinuity_ratio = discontinuities / total_frames if total_frames > 0 else 0
    kinematic_coherence = (1 - discontinuity_ratio) * 25
    
    # 3. Context Discrimination (25 points)
    # Gold frame ratio
    gold_ratio = validation_results['gold_frame_ratio']
    context_discrimination = gold_ratio * 25
    
    # 4. Hallucination Risk (25 points)
    # Inverse of low visibility ratio
    low_vis_count = 0
    total_landmarks = 0
    
    for frame in frames:
        if frame['landmarks']:
            for lm in frame['landmarks'].values():
                total_landmarks += 1
                if lm['visibility'] < 0.5:
                    low_vis_count += 1
    
    low_vis_ratio = low_vis_count / total_landmarks if total_landmarks > 0 else 0
    hallucination_risk = (1 - low_vis_ratio) * 25
    
    # Overall score
    overall_score = landmark_fidelity + kinematic_coherence + context_discrimination + hallucination_risk
    
    return {
        'landmark_fidelity': round(landmark_fidelity, 1),
        'kinematic_coherence': round(kinematic_coherence, 1),
        'context_discrimination': round(context_discrimination, 1),
        'hallucination_risk': round(hallucination_risk, 1),
        'overall_score': round(overall_score, 1),
        'detection_rate': round(detection_rate, 3),
        'discontinuity_ratio': round(discontinuity_ratio, 3),
        'gold_frame_ratio': round(gold_ratio, 3),
        'low_visibility_ratio': round(low_vis_ratio, 3)
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python module2_quality_scoring.py <pose_data.json>")
        sys.exit(1)
    
    try:
        result = validate_and_score(sys.argv[1])
        print("✓ SUCCESS!")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)