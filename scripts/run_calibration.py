#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import cv2

# Ensure project root is on sys.path so `from calibration import ...` works
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calibration.core import undistort_image
from calibration.calib_io import save_calibration_json
from calibration.utils import sorted_image_paths


def main():
    """Run calibration using OpenCV calibrateCameraExtended with ordered image paths.

    This version:
      * Keeps a single sorted list of image paths.
      * Detects chessboard corners in that order; successful detections are appended.
      * Uses cv2.calibrateCameraExtended to obtain per-view RMS errors (per_view_errors).
      * Does NOT fix distortion coefficients by default; optional flags expose common fixes.
      * Saves ordered used image paths and per-image RMSE to JSON.
    """

    ap = argparse.ArgumentParser(description="CSE5283 camera calibration (calibrateCameraExtended)")
    ap.add_argument("--images_dir", type=str, default="data/images", help="Folder with chessboard images")
    ap.add_argument("--cols", type=int, default=9, help="Inner corners across (columns)")
    ap.add_argument("--rows", type=int, default=6, help="Inner corners down (rows)")
    ap.add_argument("--square_size", type=float, default=22.0, help="Square size (mm)")
    ap.add_argument("--out_json", type=str, default="data/results/calibration.json", help="Output calibration JSON path")
    ap.add_argument("--preview_image", type=str, default=None, help="Optional image to undistort for preview")
    ap.add_argument("--preview_out", type=str, default="data/results/undistort_preview.jpg", help="Where to save side-by-side preview")
    ap.add_argument("--undistort_alpha", type=float, default=0.5, help="Alpha for getOptimalNewCameraMatrix (0..1)")
    ap.add_argument("--undistort_center_pp", action="store_true", help="Center principal point in new camera matrix for preview")
    ap.add_argument("--visualize_corners_to", type=str, default=None, help="If set, save detected-corners images to this folder")
    # Optional calibration flags (off by default)
    ap.add_argument("--zero-tangent", dest="zero_tangent", action="store_true", help="Add CALIB_ZERO_TANGENT_DIST")
    ap.add_argument("--fix-principal-point", dest="fix_principal_point", action="store_true", help="Add CALIB_FIX_PRINCIPAL_POINT")
    ap.add_argument("--fix-aspect", dest="fix_aspect", action="store_true", help="Add CALIB_FIX_ASPECT_RATIO (requires initial fx)")
    ap.add_argument("--rational", dest="rational", action="store_true", help="Add CALIB_RATIONAL_MODEL")
    ap.add_argument("--thin-prism", dest="thin_prism", action="store_true", help="Add CALIB_THIN_PRISM_MODEL")
    ap.add_argument("--tilt", dest="tilt", action="store_true", help="Add CALIB_TILTED_MODEL")
    args = ap.parse_args()

    pattern_size = (args.cols, args.rows)
    print("[1/3] Listing images...")
    image_paths = sorted_image_paths(args.images_dir)
    print(f"Found {len(image_paths)} image files in {args.images_dir}")
    if not image_paths:
        print("No images found; exiting.")
        return

    # Prepare object points template (Z=0 plane)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= float(args.square_size)  # scale to mm

    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    used_image_paths = []
    image_size = None

    print("[2/3] Detecting corners (ordered)...")
    if args.visualize_corners_to:
        os.makedirs(args.visualize_corners_to, exist_ok=True)
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                       cv2.CALIB_CB_FAST_CHECK +
                                                       cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret:
            continue
        # refine
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        if image_size is None:
            h, w = gray.shape[:2]
            image_size = (w, h)
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used_image_paths.append(p)
        if args.visualize_corners_to:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            out_name = os.path.join(args.visualize_corners_to, os.path.basename(p))
            cv2.imwrite(out_name, vis)

    if len(objpoints) < 3:
        print(f"Only detected corners in {len(objpoints)} images (need >=3). Aborting.")
        return

    print(f"Detected valid corners in {len(objpoints)} / {len(image_paths)} images.")

    # Calibration flags (none fixed by default)
    flags = 0
    if args.zero_tangent:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if args.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if args.fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if args.rational:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if args.thin_prism:
        flags |= cv2.CALIB_THIN_PRISM_MODEL
    if args.tilt:
        flags |= cv2.CALIB_TILTED_MODEL

    print("[3/3] Running calibrateCameraExtended ...")
    rms, K, dist, rvecs, tvecs, std_int, std_ext, per_view_errors = cv2.calibrateCameraExtended(
        objpoints, imgpoints, image_size, None, None, flags=flags)

    calib = {
        "K": K.tolist(),
        "distCoeffs": dist.reshape(-1).tolist(),
        "dist": dist.reshape(-1).tolist(),  # legacy key for existing loaders
        "rvecs": [rv.reshape(-1).tolist() for rv in rvecs],
        "tvecs": [tv.reshape(-1).tolist() for tv in tvecs],
        "image_paths": used_image_paths,  # ordered subset actually used
        "all_image_paths": image_paths,   # full sorted list prior to filtering
        "per_image_rmse": [float(e) for e in per_view_errors.ravel()],
        "rms": float(rms),
        "rows": int(args.rows),
        "cols": int(args.cols),
        "square_size": float(args.square_size),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "flags": int(flags),
        "std_intrinsics": [float(x) for x in std_int.reshape(-1)],
        "std_extrinsics": [float(x) for x in std_ext.reshape(-1)]
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    save_calibration_json(calib, args.out_json)

    # Summary
    print("RMS:", rms)
    print("distCoeffs:", dist.ravel())
    print("Images used:", len(used_image_paths))
    print("Saved:", args.out_json)

    # Optional undistortion preview
    if args.preview_image:
        img = cv2.imread(args.preview_image, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Preview image not found: {args.preview_image}")
            return
        K_np = K.astype(float)
        dist_np = dist.reshape(-1, 1).astype(float)
        und, newK = undistort_image(
            img, K_np, dist_np,
            balance=args.undistort_alpha,
            center_pp=args.undistort_center_pp,
        )
        side_by_side = np.hstack([img, und])
        os.makedirs(os.path.dirname(args.preview_out), exist_ok=True)
        cv2.imwrite(args.preview_out, side_by_side)
        print(f"Undistortion preview saved to {args.preview_out}")


if __name__ == "__main__":
    main()
