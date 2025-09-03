#!/usr/bin/env python3
"""
Test script for 3D camera pose visualization.

This script demonstrates the camera pose visualization by:
1. Loading calibration results
2. Converting poses to camera-in-world coordinates  
3. Creating 3D and 2D visualizations
4. Saving plots for inspection
"""

import os
import sys
import argparse

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calibration.simple_pose_viz import simple_pose_visualization


def main():
    parser = argparse.ArgumentParser(
        description="Test 3D camera pose visualization"
    )
    parser.add_argument("--calibration_json", default="data/results/calibration.json",
                       help="Path to calibration JSON file")
    parser.add_argument("--output_dir", default="data/results/pose_viz",
                       help="Output directory for visualization plots")
    parser.add_argument("--show_3d", action="store_true", default=True,
                       help="Create 3D pose visualization")
    parser.add_argument("--show_2d", action="store_true", default=True,
                       help="Create 2D trajectory plots")
    parser.add_argument("--interactive", action="store_true",
                       help="Show interactive plots (blocks execution)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("3D CAMERA POSE VISUALIZATION TEST")
    print("="*60)
    
    # Check if calibration file exists
    if not os.path.exists(args.calibration_json):
        print(f"ERROR: Calibration file not found: {args.calibration_json}")
        print("Run calibration first with:")
        print("  python scripts/run_calibration.py")
        return 1
    
    print(f"Input calibration: {args.calibration_json}")
    print(f"Output directory: {args.output_dir}")
    print(f"Show 3D: {args.show_3d}")
    print(f"Show 2D: {args.show_2d}")
    
    try:
        # Create simple visualization
        figures = simple_pose_visualization(
            args.calibration_json,
            args.output_dir,
            args.interactive
        )
        
        print(f"\nSUCCESS! Created {len(figures)} visualization(s)")
        
        if figures:
            print("Generated plots:")
            for name, fig in figures.items():
                print(f"  - {name}: {fig.get_size_inches()} inches")
        
        if args.output_dir and os.path.exists(args.output_dir):
            output_files = [f for f in os.listdir(args.output_dir) if f.endswith('.png')]
            if output_files:
                print(f"\nSaved files to {args.output_dir}:")
                for f in sorted(output_files):
                    print(f"  - {f}")
        
        # Show interactive plots if requested
        if args.interactive:
            print("\nShowing interactive plots... (close windows to continue)")
            import matplotlib.pyplot as plt
            plt.show()
        
        print(f"\nTo verify:")
        print(f"  1. Check saved plots in: {args.output_dir}")
        print(f"  2. Look for:")
        print(f"     - World frame at origin (chessboard corner)")
        print(f"     - Camera positions distributed around origin")
        print(f"     - Camera orientations pointing toward origin")
        print(f"     - Reasonable distances (depends on your setup)")
        
        return 0
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
