"""
OpenCV pose baselines: solvePnP and homography decomposition (Assignment 2)
"""
import numpy as np
import cv2

def pose_solvepnp(K: np.ndarray, dist: np.ndarray|None, XY: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate pose using OpenCV's solvePnP (planar-aware).
    K: (3,3) camera intrinsics
    dist: (5,) or None distortion
    XY: (N,2) model points (Z=0)
    uv: (N,2) image points
    Returns: (R, t)
    """
    XY = np.asarray(XY, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float32)
    obj = np.hstack([XY, np.zeros((XY.shape[0],1), dtype=np.float32)])
    img = uv
    N = XY.shape[0]
    if N >= 6:
        flag = cv2.SOLVEPNP_ITERATIVE
    elif N == 4:
        # Check if points are roughly rectangular
        dists = np.linalg.norm(XY - XY.mean(axis=0), axis=1)
        if np.std(dists) < 0.2 * np.mean(dists):
            flag = cv2.SOLVEPNP_IPPE_SQUARE
        else:
            flag = cv2.SOLVEPNP_IPPE
    else:
        flag = cv2.SOLVEPNP_IPPE
    ret, rvec, tvec = cv2.solvePnP(obj, img, K.astype(np.float32), dist if dist is not None else np.zeros(5, dtype=np.float32), flags=flag)
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.reshape(3,1).astype(np.float64)

def pose_from_h_decompose(H: np.ndarray, K: np.ndarray, XY: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose homography and select solution by cheirality.
    H: (3,3) homography
    K: (3,3) camera intrinsics
    XY: (N,2) model points (Z=0)
    Returns: (R, t)
    """
    H = np.asarray(H, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    XY = np.asarray(XY, dtype=np.float64)
    obj = np.hstack([XY, np.zeros((XY.shape[0],1))])
    Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    best_idx = -1
    best_count = -1
    for i, (R, t, n) in enumerate(zip(Rs, ts, ns)):
        pts_cam = (R @ obj.T + t.reshape(3,1))
        z = pts_cam[2]
        count = np.sum(z > 0)
        if count > best_count:
            best_count = count
            best_idx = i
    R_best = Rs[best_idx]
    t_best = ts[best_idx].reshape(3,1)
    return R_best.astype(np.float64), t_best.astype(np.float64)

# Self-check (synthetic)
if __name__ == "__main__":
    K = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=np.float64)
    theta = np.deg2rad(30)
    R_gt = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    t_gt = np.array([[0.1],[0.2],[1.0]])
    XY = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float64)
    obj = np.hstack([XY, np.zeros((4,1))])
    world_pts = R_gt @ obj.T + t_gt
    uvw = K @ world_pts
    uv = (uvw[:2] / uvw[2]).T
    H = cv2.getPerspectiveTransform(XY.astype(np.float32), uv.astype(np.float32))
    R1, t1 = pose_solvepnp(K, None, XY, uv)
    R2, t2 = pose_from_h_decompose(H, K, XY)
    def rotation_angle_diff(R1, R2):
        dR = R1 @ R2.T
        angle = np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1))
        return np.degrees(angle)
    print("solvePnP rotation diff:", rotation_angle_diff(R1, R_gt))
    print("decomposeHomographyMat rotation diff:", rotation_angle_diff(R2, R_gt))
