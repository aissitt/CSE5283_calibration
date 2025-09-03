from typing import List, Tuple
import os
import cv2
import numpy as np
from .utils import sorted_image_paths, read_gray

# OpenCV uses INNER corner counts (cols, rows)
def make_object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid *= square_size
    return grid  # (N,3)


def find_corners_in_folder(
    images_folder: str,
    pattern_size: Tuple[int, int],
    subpix_win: Tuple[int, int] = (11, 11),
    subpix_criteria: Tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3
    ),
    visualize_to: str = None,
):
    paths = sorted_image_paths(images_folder)
    if len(paths) == 0:
        raise FileNotFoundError(f"No .jpg/.jpeg images found in: {images_folder}")

    imgpoints: List[np.ndarray] = []
    used_paths: List[str] = []
    image_size = None

    cols, rows = pattern_size
    pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for p in paths:
        gray = read_gray(p)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), pattern_flags)
        if not found:
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, subpix_win, (-1, -1), subpix_criteria)
        imgpoints.append(corners_refined)  # (N,1,2)
        used_paths.append(p)

        if visualize_to:
            os.makedirs(visualize_to, exist_ok=True)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, (cols, rows), corners_refined, True)
            cv2.imwrite(os.path.join(visualize_to, os.path.basename(p)), vis)

    if len(imgpoints) < 8:
        raise RuntimeError(
            f"Corners detected in only {len(imgpoints)} images (out of {len(paths)} input images); collect more views"
        )

    return imgpoints, image_size, used_paths
