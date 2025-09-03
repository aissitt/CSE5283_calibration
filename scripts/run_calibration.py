#!/usr/bin/env python3
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import cv2

# Ensure project root is on sys.path so `from calibration import ...` works
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calibration.chessboard import find_corners_in_folder
from calibration.core import calibrate, undistort_image
from calibration.calib_io import save_calibration_json


def main():
    ap = argparse.ArgumentParser(description="CSE5283 camera calibration (Ari).")
    ap.add_argument("--images_dir", type=str, default="data/images", help="Folder with .jpeg chessboard images")
    ap.add_argument("--cols", type=int, default=9, help="Inner corners across (columns)")
    ap.add_argument("--rows", type=int, default=6, help="Inner corners down (rows)")
    ap.add_argument("--square_size", type=float, default=25.0, help="Square size in millimeters")
    ap.add_argument("--visualize_corners_to", type=str, default="data/results/corners", help="Save drawn corners here")
    ap.add_argument("--out_json", type=str, default="data/results/calibration.json", help="Output calibration JSON")
    ap.add_argument("--preview_image", type=str, default=None, help="One image to undistort (saves side-by-side)")
    ap.add_argument("--preview_out", type=str, default="data/results/undistort_preview.jpg")
    ap.add_argument("--undistort_alpha", type=float, default=0.5, help="Alpha (balance) for getOptimalNewCameraMatrix: 0.0..1.0")
    ap.add_argument("--undistort_center_pp", action="store_true", help="Center principal point in new camera matrix")
    ap.add_argument("--use_rational_model", action="store_true", help="Enable CALIB_RATIONAL_MODEL")
    args = ap.parse_args()

    pattern_size = (args.cols, args.rows)

    print("[1/3] Detecting corners...")
    # report how many input images we found
    from calibration.utils import sorted_image_paths
    all_paths = sorted_image_paths(args.images_dir)
    print(f"Found {len(all_paths)} image files in {args.images_dir}")

    imgpoints, image_size, used_paths = find_corners_in_folder(
        args.images_dir,
        pattern_size,
        visualize_to=args.visualize_corners_to,
    )
    print(f"Detected corners in {len(imgpoints)} images; overlays saved for {len(used_paths)} files")

    flags = 0
    if args.use_rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL

    print("[2/3] Calibrating...")
    calib = calibrate(
        pattern_size=pattern_size,
        square_size=args.square_size,
        imgpoints=imgpoints,
        image_size=image_size,
        flags=flags
    )
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    save_calibration_json(calib, args.out_json)
    print(f"Saved calibration to {args.out_json}")
    print(f"RMS reprojection error: {calib['rms']:.4f}")
    if 'per_image_reproj_rmse' in calib:
        print(f"Mean per-image RMSE: {np.mean(calib['per_image_reproj_rmse']):.4f}")

    if args.preview_image:
        print("[3/3] Creating undistortion preview...")
        img = cv2.imread(args.preview_image, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"preview_image not found: {args.preview_image}")
        K = np.array(calib["K"], dtype=float)
        dist = np.array(calib["dist"], dtype=float).reshape(-1, 1)
        und, newK = undistort_image(
            img, K, dist,
            balance=args.undistort_alpha,
            center_pp=args.undistort_center_pp,
        )
        side_by_side = np.hstack([img, und])
        os.makedirs(os.path.dirname(args.preview_out), exist_ok=True)
        cv2.imwrite(args.preview_out, side_by_side)
        print(f"Undistortion preview saved to {args.preview_out}")


if __name__ == "__main__":
    main()
