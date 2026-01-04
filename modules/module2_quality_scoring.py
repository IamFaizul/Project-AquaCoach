"""
MODULE 2: Quality Scoring (Simplified - Ryan's Approach)
Validates frames by visibility + completeness, then scores

Input: pose_data.json from Module 1
Output: Quality score + list of gold frames
"""

import json
from pathlib import Path


# ==================== CONFIGURATION ====================

CONFIG = {
    # Validation thresholds
    'visibility_threshold': 0.6,
    'min_landmarks_required': 7,
    
    # Ryan's scoring cliffs
    'null_frame_cliff': 0.2,
    'low_vis_cliff': 0.4,
}

MAX_SCORE = 25  # Per component


# ==================== VALIDATION (SIMPLIFIED) ====================

def validate_frame(frame_landmarks):
    """
    Simple 2-layer validation:
    1. Visibility check (≥0.6)
    2. Completeness check (≥7 landmarks)
    
    Returns: True if frame is GOLD, False otherwise
    """
    if not frame_landmarks:
        return False
    
    # Count landmarks with good visibility
    good_landmarks = 0
    
    for landmark in frame_landmarks.values():
        if landmark['visibility'] >= CONFIG['visibility_threshold']:
            good_landmarks += 1
    
    # Frame passes if it has enough good landmarks
    return good_landmarks >= CONFIG['min_landmarks_required']


def find_gold_frames(frames_data):
    """
    Apply validation to find gold frames
    
    Returns: Dict with validation results
    """
    total_frames = len(frames_data)
    
    # Track results
    passed_visibility = 0
    gold_frames = []
    
    print("\nValidating frames (2-layer approach)...")
    
    for frame in frames_data:
        frame_num = frame['frame_number']
        landmarks = frame.get('landmarks')
        
        if not landmarks:
            continue
        
        # Layer 1: Count landmarks with good visibility
        good_landmarks = sum(
            1 for lm in landmarks.values() 
            if lm['visibility'] >= CONFIG['visibility_threshold']
        )
        
        if good_landmarks > 0:
            passed_visibility += 1
        
        # Layer 2: Check if enough good landmarks
        if good_landmarks >= CONFIG['min_landmarks_required']:
            gold_frames.append(frame_num)
    
    print(f"✓ Validation complete")
    
    return {
        'total_frames': total_frames,
        'passed_visibility': passed_visibility,
        'gold_frames': gold_frames,
        'gold_frame_count': len(gold_frames),
        'gold_frame_percentage': len(gold_frames) / total_frames if total_frames > 0 else 0
    }


# ==================== QUALITY SCORING (RYAN'S FORMULA) ====================

def calculate_quality_score(pose_data, validation_results):
    """
    Calculate overall quality score 0-100 using Ryan's formula
    
    Components:
    1. Landmark Fidelity & Continuity (0-25)
    2. Kinematic Coherence (0-25)
    3. Context Discrimination (0-25)
    4. Hallucination Risk Control (0-25)
    """
    stats = pose_data['statistics']
    validation = validation_results
    
    # Extract key ratios
    null_ratio = stats['null_frame_ratio']
    low_vis_ratio = stats['low_visibility_ratio']
    disc_ratio = stats['discontinuity_ratio']
    gold_ratio = validation['gold_frame_percentage']
    
    # COMPONENT 1: Landmark Fidelity (0-25)
    landmark_fidelity = MAX_SCORE * (1 - null_ratio) * (1 - low_vis_ratio)
    
    # Cliff penalties
    if null_ratio > CONFIG['null_frame_cliff']:
        landmark_fidelity *= 0.6
    if low_vis_ratio > CONFIG['low_vis_cliff']:
        landmark_fidelity *= 0.5
    
    # COMPONENT 2: Kinematic Coherence (0-25)
    kinematic_coherence = MAX_SCORE * (1 - disc_ratio) * (1 - low_vis_ratio)
    
    # Cliff penalties
    if null_ratio > CONFIG['null_frame_cliff']:
        kinematic_coherence *= 0.6
    if low_vis_ratio > CONFIG['low_vis_cliff']:
        kinematic_coherence *= 0.5
    
    # COMPONENT 3: Context Discrimination (0-25)
    context_discrimination = MAX_SCORE * gold_ratio
    
    # COMPONENT 4: Hallucination Risk Control (0-25)
    hallucination_control = MAX_SCORE * (1 - low_vis_ratio) * (1 - disc_ratio)
    hallucination_control *= 0.6  # Skepticism multiplier
    
    # Cliff penalties
    if null_ratio > CONFIG['null_frame_cliff']:
        hallucination_control *= 0.7
    if low_vis_ratio > CONFIG['low_vis_cliff']:
        hallucination_control *= 0.7
    
    # Clamp to 0-25 range
    landmark_fidelity = max(0, min(MAX_SCORE, landmark_fidelity))
    kinematic_coherence = max(0, min(MAX_SCORE, kinematic_coherence))
    context_discrimination = max(0, min(MAX_SCORE, context_discrimination))
    hallucination_control = max(0, min(MAX_SCORE, hallucination_control))
    
    # Total score
    overall_score = (
        landmark_fidelity +
        kinematic_coherence +
        context_discrimination +
        hallucination_control
    )
    
    return {
        'landmark_fidelity_continuity': round(landmark_fidelity, 1),
        'kinematic_coherence': round(kinematic_coherence, 1),
        'context_discrimination': round(context_discrimination, 1),
        'hallucination_risk_control': round(hallucination_control, 1),
        'overall_score': round(overall_score, 1)
    }


# ==================== MAIN FUNCTION ====================

def score_video_quality(pose_data_path, output_folder='output'):
    """
    Main function: Load pose data, validate, score
    
    Args:
        pose_data_path: Path to pose_data.json from Module 1
        output_folder: Where to save results
    
    Returns:
        Path to quality scoring output JSON
    """
    
    # Load pose data
    print(f"\n{'='*60}")
    print(f"LOADING POSE DATA")
    print(f"{'='*60}\n")
    
    with open(pose_data_path, 'r') as f:
        pose_data = json.load(f)
    
    video_name = Path(pose_data['video_info']['source']).stem
    frames_data = pose_data['frames']
    
    print(f"Video: {video_name}")
    print(f"Total frames: {len(frames_data)}")
    
    # Apply simplified validation
    print(f"\n{'='*60}")
    print(f"VALIDATION (2-LAYER APPROACH)")
    print(f"{'='*60}")
    
    validation_results = find_gold_frames(frames_data)
    
    # Display validation results
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS:")
    print(f"  Total frames: {validation_results['total_frames']}")
    print(f"  Frames with good visibility: {validation_results['passed_visibility']} "
          f"({validation_results['passed_visibility']/validation_results['total_frames']*100:.1f}%)")
    print(f"\n  🏆 GOLD FRAMES: {validation_results['gold_frame_count']} "
          f"({validation_results['gold_frame_percentage']*100:.1f}%)")
    print(f"{'='*60}")
    
    # Calculate quality score
    print(f"\n{'='*60}")
    print(f"QUALITY SCORING")
    print(f"{'='*60}\n")
    
    quality_scores = calculate_quality_score(pose_data, validation_results)
    
    print(f"Scoring Breakdown:")
    print(f"  Landmark Fidelity: {quality_scores['landmark_fidelity_continuity']}/25")
    print(f"  Kinematic Coherence: {quality_scores['kinematic_coherence']}/25")
    print(f"  Context Discrimination: {quality_scores['context_discrimination']}/25")
    print(f"  Hallucination Risk: {quality_scores['hallucination_risk_control']}/25")
    print(f"\n  📊 OVERALL SCORE: {quality_scores['overall_score']}/100")
    
    # Determine if video is usable
    is_usable = quality_scores['overall_score'] >= 70
    print(f"\n  {'✅' if is_usable else '⚠️'} Video Status: {'USABLE' if is_usable else 'NEEDS IMPROVEMENT'}")
    print(f"{'='*60}\n")
    
    # Save results
    Path(output_folder).mkdir(exist_ok=True)
    
    output_data = {
        'video_name': video_name,
        'overall_quality_score': quality_scores['overall_score'],
        'is_video_usable': is_usable,
        'quality_breakdown': quality_scores,
        'validation_results': validation_results,
        'configuration': CONFIG
    }
    
    output_path = Path(output_folder) / f"{video_name}_quality_report.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ SAVED: {output_path}\n")
    
    return str(output_path)


# ==================== CLI ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python module2_quality_scoring.py <pose_data.json>")
        sys.exit(1)
    
    pose_data_path = sys.argv[1]
    
    try:
        result = score_video_quality(pose_data_path)
        print(f"\n✓ SUCCESS! Quality report: {result}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()