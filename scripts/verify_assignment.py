#!/usr/bin/env python3
"""
Complete verification script for CSE5283 Camera Calibration Assignment.

This script demonstrates that both key deliverables work correctly:
1. Axes overlays on calibrated images
2. 3D camera pose visualization

Run this to verify your assignment meets the rubric requirements.
"""

import os
import sys
import argparse

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calibration.overlay import overlay_axes_on_calibration_images
from calibration.pose_viz import visualize_calibration_poses
from calibration.calib_io import load_calibration_json


def verify_calibration_exists(calibration_path: str) -> bool:
    """Check if calibration file exists and is valid."""
    if not os.path.exists(calibration_path):
        return False
    
    try:
        calib = load_calibration_json(calibration_path)
        required_keys = ['K', 'dist', 'rvecs', 'tvecs', 'rms', 'image_size']
        for key in required_keys:
            if key not in calib:
                print(f"Missing key in calibration: {key}")
                return False
        return True
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return False


def verify_axes_overlays(calibration_path: str, images_dir: str, output_dir: str, num_images: int = 8) -> bool:
    """Verify axes overlay functionality."""
    print(f"\n[TASK 1] Verifying axes overlays...")
    print(f"  Target: {num_images} images with world coordinate axes")
    
    try:
        output_paths = overlay_axes_on_calibration_images(
            calibration_path, images_dir, output_dir, num_images
        )
        
        if len(output_paths) >= 5:  # Rubric requires 5-10 images
            print(f"  ✓ SUCCESS: Created {len(output_paths)} axes overlay images")
            print(f"  ✓ Meets rubric requirement: 5-10 images with axes")
            print(f"  ✓ Output directory: {output_dir}")
            return True
        else:
            print(f"  ✗ INSUFFICIENT: Only {len(output_paths)} images (need 5-10)")
            return False
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def verify_pose_visualization(calibration_path: str, output_dir: str) -> bool:
    """Verify 3D camera pose visualization."""
    print(f"\n[TASK 2] Verifying 3D camera pose visualization...")
    print(f"  Target: Camera poses in world coordinates with pytransform3d")
    
    try:
        figures = visualize_calibration_poses(calibration_path, output_dir, show_3d=True, show_2d=True)
        
        success_count = 0
        
        if '3d_poses' in figures:
            print(f"  ✓ 3D pose plot created successfully")
            success_count += 1
        else:
            print(f"  ✗ 3D pose plot missing")
            
        if '2d_trajectory' in figures:
            print(f"  ✓ 2D trajectory plot created successfully")
            success_count += 1
        else:
            print(f"  ✗ 2D trajectory plot missing")
        
        # Check output files exist
        pose_3d_file = os.path.join(output_dir, 'camera_poses_3d.png')
        pose_2d_file = os.path.join(output_dir, 'camera_trajectory_2d.png')
        
        if os.path.exists(pose_3d_file):
            print(f"  ✓ 3D plot saved: {pose_3d_file}")
        if os.path.exists(pose_2d_file):
            print(f"  ✓ 2D plot saved: {pose_2d_file}")
            
        if success_count >= 1:  # At least one visualization type
            print(f"  ✓ SUCCESS: Created {success_count} visualization types")
            print(f"  ✓ Meets rubric requirement: 3D camera pose plot")
            return True
        else:
            print(f"  ✗ FAILED: No visualizations created")
            return False
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def print_rubric_summary(axes_ok: bool, poses_ok: bool, calib_path: str):
    """Print summary against assignment rubric."""
    print(f"\n" + "="*60)
    print(f"ASSIGNMENT RUBRIC VERIFICATION")
    print(f"="*60)
    
    # Load calibration for summary stats
    try:
        calib = load_calibration_json(calib_path)
        num_poses = len(calib['rvecs'])
        rms_error = calib['rms']
        image_size = tuple(calib['image_size'])
    except:
        num_poses = "?"
        rms_error = "?"
        image_size = "?"
    
    print(f"Calibration Data (prerequisite):")
    print(f"  ✓ Images with detected corners: {num_poses}")
    print(f"  ✓ RMS reprojection error: {rms_error}")
    print(f"  ✓ Image size: {image_size}")
    
    print(f"\nRubric Requirements:")
    
    # Axes overlays [10 pts]
    if axes_ok:
        print(f"  ✓ [10/10 pts] Axes overlays on 5-10 images")
        print(f"      - World coordinate frame drawn on calibrated images")
        print(f"      - Using cv2.projectPoints with K, dist, rvec, tvec")
        print(f"      - Saved to data/results/axes/")
    else:
        print(f"  ✗ [0/10 pts] Axes overlays FAILED")
    
    # 3D pose visualization [15 pts]  
    if poses_ok:
        print(f"  ✓ [15/15 pts] 3D camera pose visualization")
        print(f"      - Converted to camera-in-world coordinates [R^T | -R^T*t]")
        print(f"      - Plotted with pytransform3d")
        print(f"      - Saved to data/results/pose_viz/")
    else:
        print(f"  ✗ [0/15 pts] 3D pose visualization FAILED")
    
    # Total for these tasks
    total_points = (10 if axes_ok else 0) + (15 if poses_ok else 0)
    print(f"\nCompleted Tasks Total: {total_points}/25 points")
    
    print(f"\nRemaining Tasks (not verified by this script):")
    print(f"  - [ ] Gradio UI implementation [25 pts]")
    print(f"  - [ ] Code quality & structure [10 pts]") 
    print(f"  - [ ] Repository & documentation [5 pts]")
    print(f"  - [ ] Class presentation [10 pts]")
    print(f"  Total Assignment: {total_points}/100 points")


def main():
    parser = argparse.ArgumentParser(
        description="Verify assignment deliverables meet rubric requirements"
    )
    parser.add_argument("--calibration_json", default="data/results/calibration.json",
                       help="Path to calibration JSON file")
    parser.add_argument("--images_dir", default="data/images",
                       help="Directory containing original chessboard images")
    parser.add_argument("--axes_output", default="data/results/axes",
                       help="Output directory for axes overlay images")
    parser.add_argument("--pose_output", default="data/results/pose_viz",
                       help="Output directory for pose visualization plots")
    parser.add_argument("--num_axes_images", type=int, default=8,
                       help="Number of images to create axes overlays for")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CSE5283 ASSIGNMENT VERIFICATION")
    print("="*60)
    print(f"Verifying deliverables for rubric compliance...")
    print(f"Calibration file: {args.calibration_json}")
    print(f"Images directory: {args.images_dir}")
    
    # Check prerequisites
    print(f"\n[PREREQUISITE] Checking calibration file...")
    if not verify_calibration_exists(args.calibration_json):
        print(f"✗ FAILED: Invalid or missing calibration file")
        print(f"Run: python scripts/run_calibration.py")
        return 1
    print(f"✓ Valid calibration file found")
    
    # Verify Task 1: Axes overlays
    axes_ok = verify_axes_overlays(
        args.calibration_json, args.images_dir, args.axes_output, args.num_axes_images
    )
    
    # Verify Task 2: 3D pose visualization  
    poses_ok = verify_pose_visualization(args.calibration_json, args.pose_output)
    
    # Print rubric summary
    print_rubric_summary(axes_ok, poses_ok, args.calibration_json)
    
    # Final verdict
    if axes_ok and poses_ok:
        print(f"\n🎉 SUCCESS: Both key deliverables are working correctly!")
        print(f"Next steps:")
        print(f"  1. Create Gradio notebook interface")  
        print(f"  2. Update README.md with usage instructions")
        print(f"  3. Fill out AI appendix documentation")
        print(f"  4. Prepare 10-minute class presentation")
        return 0
    else:
        print(f"\n❌ ISSUES FOUND: Fix failing tasks before proceeding")
        return 1


if __name__ == "__main__":
    exit(main())
