"""
Pose from Homography (Prince §15.4.1)

Unit test:
>>> import numpy as np
>>> from planar_pose.homography import dlt_homography
>>> from planar_pose.pose_from_homography import pose_from_homography
>>> def rotation_angle_diff(R1, R2):
...     dR = R1 @ R2.T
...     angle = np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1))
...     return np.degrees(angle)
>>> def translation_dir_angle(t1, t2):
...     t1 = t1.flatten() / np.linalg.norm(t1)
...     t2 = t2.flatten() / np.linalg.norm(t2)
...     dot = np.clip(np.dot(t1, t2), -1, 1)
...     return np.degrees(np.arccos(dot))
>>> K = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=np.float64)
>>> theta = np.deg2rad(30)
>>> R = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
>>> t = np.array([[0.1],[0.2],[1.0]])
>>> XY = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float64)
>>> model_pts = np.hstack([XY, np.zeros((4,1))])
>>> world_pts = R @ model_pts.T + t
>>> uvw = K @ world_pts
>>> uv = (uvw[:2] / uvw[2]).T
>>> H = dlt_homography(XY, uv)
>>> R_hat, t_hat = pose_from_homography(H, K)
>>> assert rotation_angle_diff(R_hat, R) < 1.0
>>> assert translation_dir_angle(t_hat, t) < 3.0
"""
import numpy as np

def pose_from_homography(H: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover pose (R, t) from homography and intrinsics.
    H: (3,3) homography
    K: (3,3) camera intrinsics
    Returns: (R, t)
    """
    H = np.asarray(H, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    H_tilde = np.linalg.inv(K) @ H
    h1 = H_tilde[:,0]
    h2 = H_tilde[:,1]
    h3 = H_tilde[:,2]
    lam = 1.0 / np.mean([np.linalg.norm(h1), np.linalg.norm(h2)])
    r1 = lam * h1
    r2 = lam * h2
    r3 = np.cross(r1, r2)
    R_approx = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:,2] *= -1
    t = (lam * h3).reshape(3,1)
    return R.astype(np.float64), t.astype(np.float64)
