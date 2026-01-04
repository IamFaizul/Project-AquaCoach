"""
FINAL PROCESSING SCRIPT
Process all swimming videos with optimized pipeline
"""

import subprocess
import sys
from pathlib import Path


VIDEOS = [
    "data/Clem BW.mov",
    "data/JP UW.mov",
    "data/PS UW.mov",
    "data/Tomaso UW.mov"
]


def run_module(module_name, *args):
    """Run a module with given arguments"""
    cmd = [sys.executable, f"{module_name}.py"] + list(args)
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def process_video(video_path, create_video=True):
    """Process one video through all modules"""
    
    if not Path(video_path).exists():
        print(f"✗ Video not found: {video_path}")
        return False
    
    video_name = Path(video_path).stem
    
    print(f"\n{'='*70}")
    print(f"PROCESSING: {video_name}")
    print(f"{'='*70}\n")
    
    # Module 1: Optimized Pose Detection
    print("▶ MODULE 1: Optimized Pose Detection")
    print("-" * 70)
    if not run_module("module1_pose_optimized", video_path):
        return False
    
    pose_data_path = f"output/{video_name}_pose_data.json"
    
    # Module 2: Quality Scoring
    print("\n▶ MODULE 2: Quality Scoring")
    print("-" * 70)
    if not run_module("module2_quality_scoring", pose_data_path):
        return False
    
    quality_report_path = f"output/{video_name}_quality_report.json"
    
    # Module 3: Angle Metrics
    print("\n▶ MODULE 3: Angle Metrics")
    print("-" * 70)
    if not run_module("module3_angle_metrics", pose_data_path, quality_report_path):
        return False
    
    metrics_path = f"output/{video_name}_metrics.json"
    
    # Module 4: Visualization (optional)
    if create_video:
        print("\n▶ MODULE 4: Creating Annotated Video")
        print("-" * 70)
        if not run_module("module4_visualization", video_path, pose_data_path, quality_report_path, metrics_path):
            print("⚠ Warning: Visualization failed, but continuing...")
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETED: {video_name}")
    print(f"{'='*70}\n")
    
    return True


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print("AQUACOACH FINAL - OPTIMIZED SWIMMING ANALYSIS")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Optimized pose detection (Model Complexity 2)")
    print("  ✓ 5-frame smoothing + outlier removal")
    print("  ✓ Bilateral filtering")
    print("  ✓ High-accuracy angle measurements")
    print("  ✓ Quality scoring with gold frames")
    print("  ✓ Annotated videos with skeleton overlay")
    print(f"\nProcessing {len(VIDEOS)} videos:")
    for i, video in enumerate(VIDEOS, 1):
        print(f"  {i}. {Path(video).name}")
    
    print("\nCreate annotated videos?")
    print("  [Y] Yes, create annotated videos (~20-25 min total)")
    print("  [N] No, just extract data (~10-15 min total)")
    
    choice = input("\nChoice [Y/n]: ").strip().lower()
    create_videos = choice != 'n'
    
    if create_videos:
        print("\n✓ Will create annotated videos")
    else:
        print("\n✓ Skipping annotated videos (data only)")
    
    # Process each video
    results = []
    
    for video in VIDEOS:
        success = process_video(video, create_videos)
        results.append((Path(video).name, success))
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*70 + "\n")
    
    for video_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {video_name}")
    
    successful = sum(1 for _, s in results if s)
    
    print(f"\n  Total: {successful}/{len(VIDEOS)} videos processed successfully")
    
    print(f"\n  Output files per video:")
    print(f"    - pose_data.json (optimized pose landmarks)")
    print(f"    - quality_report.json (validation & quality score)")
    print(f"    - metrics.json (angle measurements)")
    if create_videos:
        print(f"    - annotated.mp4 (skeleton overlay video)")
    
    print("\n" + "="*70 + "\n")
    
    if successful == len(VIDEOS):
        print("✓ All videos processed successfully!")
        print("\nYou can now:")
        print("  1. Check output/*.json files for detailed metrics")
        print("  2. Watch output/*_annotated.mp4 videos")
        print("  3. Show results to your employer")
        return 0
    else:
        print("⚠ Some videos failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)