"""
Comparison metrics for pose estimation (Assignment 2)

>>> import numpy as np
>>> R1 = np.eye(3)
>>> theta = np.deg2rad(10)
>>> R2 = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
>>> bool(abs(rotation_angle_diff(R1, R2)) < 10.1)
True
>>> t1 = np.array([[1],[0],[0]])
>>> t2 = np.array([[0],[1],[0]])
>>> bool(abs(translation_dir_angle(t1, t2) - 90) < 1)
True
>>> K = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=np.float64)
>>> R = np.eye(3)
>>> t = np.array([[0],[0],[1]])
>>> XY = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
>>> uv = np.array([[320,240],[1120,240],[1120,1040],[320,1040]], dtype=np.float32)
>>> err = reprojection_error(K, None, R, t, XY, uv)
>>> err['mean'] < 1e-6
True
"""
import numpy as np
import cv2

def rotation_angle_diff(R1, R2) -> float:
    """
    Compute rotation angle difference in degrees.
    """
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    dR = R2.T @ R1
    angle = np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1))
    return np.degrees(angle)

def translation_dir_angle(t1, t2) -> float:
    """
    Angle between translation directions in degrees.
    """
    t1 = np.asarray(t1).flatten()
    t2 = np.asarray(t2).flatten()
    n1 = np.linalg.norm(t1)
    n2 = np.linalg.norm(t2)
    if n1 == 0 or n2 == 0:
        return 0.0
    t1u = t1 / n1
    t2u = t2 / n2
    dot = np.clip(np.dot(t1u, t2u), -1, 1)
    return np.degrees(np.arccos(dot))

def reprojection_error(K, dist, R, t, XY, uv) -> dict:
    """
    Compute reprojection error statistics.
    """
    XY = np.asarray(XY, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float32)
    obj_pts = np.hstack([XY, np.zeros((XY.shape[0],1))])
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.astype(np.float32)
    uv_proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    uv_proj = uv_proj.squeeze()
    errors = np.linalg.norm(uv_proj - uv, axis=1)
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
        "per_point": errors
    }
