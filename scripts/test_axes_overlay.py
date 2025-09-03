#!/usr/bin/env python3
"""
Test script for axes overlay functionality.

This script demonstrates and tests the axes overlay feature by:
1. Running camera calibration if needed
2. Overlaying coordinate axes on calibrated images
3. Saving results for visual inspection
"""

import os
import sys
import argparse

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calibration.overlay import overlay_axes_on_calibration_images
from calibration.chessboard import find_corners_in_folder
from calibration.core import calibrate
from calibration.calib_io import save_calibration_json


def run_calibration_if_needed(
    images_dir: str,
    pattern_size: tuple,
    square_size: float,
    output_json: str
) -> bool:
    """Run calibration if output JSON doesn't exist."""
    if os.path.exists(output_json):
        print(f"Calibration file already exists: {output_json}")
        return True
    
    print("Running calibration...")
    
    # Find corners
    imgpoints, image_size, used_paths = find_corners_in_folder(
        images_dir, pattern_size, visualize_to=None
    )
    
    if len(imgpoints) < 8:
        print(f"ERROR: Only {len(imgpoints)} images with detected corners. Need at least 8.")
        return False
    
    # Calibrate
    calib = calibrate(pattern_size, square_size, imgpoints, image_size)
    
    # Save results
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    save_calibration_json(calib, output_json)
    
    print(f"Calibration completed. RMS error: {calib['rms']:.4f}")
    print(f"Saved to: {output_json}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test axes overlay functionality on calibrated images"
    )
    parser.add_argument("--images_dir", default="data/images", 
                       help="Directory containing chessboard images")
    parser.add_argument("--cols", type=int, default=9, 
                       help="Inner corners across (columns)")
    parser.add_argument("--rows", type=int, default=6, 
                       help="Inner corners down (rows)")
    parser.add_argument("--square_size", type=float, default=25.0,
                       help="Square size in millimeters")
    parser.add_argument("--calibration_json", default="data/results/calibration.json",
                       help="Path to calibration JSON file")
    parser.add_argument("--output_dir", default="data/results/axes",
                       help="Output directory for images with axes")
    parser.add_argument("--max_images", type=int, default=10,
                       help="Maximum number of images to process")
    # Axis length is now derived from calibration's square_size by default
    # You can still override by passing --axis_length_mm
    parser.add_argument("--axis_length_mm", type=float, default=None,
                       help="Optional: length of axes in millimeters (default: ~half board dimension)")
    parser.add_argument("--origin_mode", type=str, default="corner", choices=["corner", "center", "board_center", "image_center", "always_center"],
                       help="Where to place the world origin: corner, center, board_center, image_center, or always_center")
    parser.add_argument("--thickness", type=int, default=None,
                       help="Optional: line thickness for arrows (auto if omitted)")
    
    args = parser.parse_args()
    
    pattern_size = (args.cols, args.rows)
    
    print("="*60)
    print("AXES OVERLAY TEST")
    print("="*60)
    
    # Step 1: Ensure calibration exists
    print("\n[1/2] Checking calibration...")
    if not run_calibration_if_needed(
        args.images_dir, pattern_size, args.square_size, args.calibration_json
    ):
        print("FAILED: Could not obtain calibration")
        return 1
    
    # Step 2: Generate axes overlays
    print(f"\n[2/2] Generating axes overlays...")
    print(f"Input images: {args.images_dir}")
    print(f"Calibration: {args.calibration_json}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max images: {args.max_images}")
    if args.axis_length_mm is not None:
        print(f"Axis length override: {args.axis_length_mm} mm")
    else:
        print(f"Axis length: default (10 * square_size = 250 mm)")
    print(f"Origin mode: {args.origin_mode}")
    print(f"Thickness: {'auto' if args.thickness is None else args.thickness}")
    
    try:
        output_paths = overlay_axes_on_calibration_images(
            args.calibration_json,
            args.images_dir,
            args.output_dir,
            args.max_images,
            args.axis_length_mm,
            origin_mode=args.origin_mode,
            thickness=args.thickness
        )
        
        print(f"\nSUCCESS! Created {len(output_paths)} images with axes overlays")
        print(f"Results saved to: {args.output_dir}")
        print("\nSample outputs:")
        for i, path in enumerate(output_paths[:3]):
            print(f"  {i+1}. {os.path.basename(path)}")
        if len(output_paths) > 3:
            print(f"  ... and {len(output_paths) - 3} more")
            
        print(f"\nTo verify: open images in {args.output_dir}")
        print("Look for:")
        print("  - Red line: X-axis (along chessboard rows)")
        print("  - Green line: Y-axis (along chessboard columns)")  
        print("  - Blue line: Z-axis (perpendicular to chessboard)")
        print("  - Black dot: Origin (corner of chessboard)")
        print(f"\nOrigin modes:")
        print(f"  - corner: World origin at chessboard corner")
        print(f"  - center: World origin at chessboard center") 
        print(f"  - board_center: Same as center")
        print(f"  - image_center: Always at image center (demo)")
        print(f"  - always_center: Always at image center (guaranteed visible)")
        
        return 0
        
    except Exception as e:
        print(f"FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
