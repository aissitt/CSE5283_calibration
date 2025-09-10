"""
Homography estimation via Normalized DLT and RANSAC (Assignment 2)
"""
import numpy as np
import cv2

def _normalize_points(pts: np.ndarray):
    """Hartley normalization: returns normalized points and similarity transform."""
    pts = np.asarray(pts, dtype=np.float64)
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    scale = np.sqrt(2) / std if np.all(std > 0) else np.ones_like(std)
    T = np.array([
        [scale[0], 0, -scale[0]*mean[0]],
        [0, scale[1], -scale[1]*mean[1]],
        [0, 0, 1]
    ])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T

def dlt_homography(XY: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """
    Normalized DLT for homography estimation.
    XY: (N,2) model points (Z=0)
    uv: (N,2) image points
    Returns: H (3x3)
    """
    XY = np.asarray(XY, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    if XY.shape[0] < 4 or uv.shape[0] < 4:
        raise ValueError("At least 4 points required.")
    if XY.shape[1] != 2 or uv.shape[1] != 2:
        raise ValueError("Points must be 2D.")
    # Normalize
    XY_norm, Tm = _normalize_points(XY)
    uv_norm, Ti = _normalize_points(uv)
    N = XY.shape[0]
    A = []
    for i in range(N):
        X, Y = XY_norm[i]
        u, v = uv_norm[i]
        A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
        A.append([X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u])
    A = np.array(A)
    # SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    Hn = h.reshape(3,3)
    # Denormalize
    H = np.linalg.inv(Ti) @ Hn @ Tm
    H /= H[2,2]
    return H

def dlt_homography_ransac(XY: np.ndarray, uv: np.ndarray, thresh_px: float=3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Homography with RANSAC. Returns (H, inliers_mask_bool)
    """
    XY = np.asarray(XY, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    if XY.shape[0] < 4 or uv.shape[0] < 4:
        raise ValueError("At least 4 points required.")
    H, mask = cv2.findHomography(XY, uv, method=cv2.RANSAC, ransacReprojThreshold=thresh_px)
    if H is None:
        H = dlt_homography(XY, uv)
        mask = np.ones((XY.shape[0], 1), dtype=np.uint8)
    mask_bool = mask.astype(bool).flatten()
    H = H.astype(np.float64)
    H /= H[2,2]
    return H, mask_bool

# Quick self-test
if __name__ == "__main__":
    # Synthetic square
    XY = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float64)
    # Simulate H: scale=2, rotate=45deg, translate=(5,10)
    theta = np.deg2rad(45)
    S = 2.0
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([5,10])
    uv = (S * (R @ XY.T).T) + t
    H_est = dlt_homography(XY, uv)
    uv_proj = cv2.perspectiveTransform(np.hstack([XY, np.zeros((4,1))]).reshape(-1,1,3), H_est)
    print("H_est:\n", H_est)
    print("uv_proj:\n", uv_proj.squeeze())
