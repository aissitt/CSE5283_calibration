from typing import Dict, List, Tuple
import cv2
import numpy as np
from .chessboard import make_object_points


def calibrate(
    pattern_size: Tuple[int, int],
    square_size: float,
    imgpoints: List[np.ndarray],
    image_size: Tuple[int, int],
    flags: int = 0,
) -> Dict:
    objp = make_object_points(pattern_size, square_size)
    objpoints = [objp.copy() for _ in imgpoints]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )

    per_image = compute_per_image_errors(objpoints, imgpoints, rvecs, tvecs, K, dist)

    return {
        "rms": float(rms),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "rvecs": [rv.reshape(-1).tolist() for rv in rvecs],
        "tvecs": [tv.reshape(-1).tolist() for tv in tvecs],
        "per_image_reproj_rmse": per_image,
        "pattern_size": [int(pattern_size[0]), int(pattern_size[1])],
        "square_size": float(square_size),
        "flags": int(flags),
    }


def compute_per_image_errors(
    objpoints, imgpoints, rvecs, tvecs, K, dist
) -> List[float]:
    errors = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        err = cv2.norm(imgp, proj, cv2.NORM_L2) / np.sqrt(len(objp))
        errors.append(float(err))
    return errors


def undistort_image(img_bgr, K, dist, balance: float = 0.5, center_pp: bool = True):
    """Undistort an image using the provided calibration.

    Args:
        img_bgr: input BGR image.
        K: camera matrix.
        dist: distortion coefficients.
        balance: alpha parameter passed to cv2.getOptimalNewCameraMatrix (0.0..1.0).
                 0.0 crops to the valid region (zoomed), 1.0 keeps full FOV (may add borders).
        center_pp: whether to center the principal point in the new camera matrix.

    Returns:
        (undistorted_image, new_camera_matrix)
    """
    h, w = img_bgr.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(
        K, dist, (w, h), alpha=float(balance), newImgSize=(w, h), centerPrincipalPoint=bool(center_pp)
    )
    und = cv2.undistort(img_bgr, K, dist, None, newK)
    return und, newK
