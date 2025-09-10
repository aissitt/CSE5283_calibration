"""
Overlay points/axes and plot 3D camera poses (Assignment 2)
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def overlay_points_and_axes(img_bgr, K, dist, R, t, XY, axis_len_m=0.05, point_color=(255,255,0)) -> np.ndarray:
    """
    Overlay reprojected points and axes on image.
    img_bgr: input image
    K: camera intrinsics
    dist: distortion or None
    R: (3,3) rotation
    t: (3,1) translation
    XY: (N,2) model points
    axis_len_m: length of axes in meters
    point_color: BGR tuple
    Returns: image copy with overlays
    """
    img = img_bgr.copy()
    obj_pts = np.hstack([XY, np.zeros((XY.shape[0],1))]).astype(np.float32)
    axes_pts = np.array([[0,0,0],[axis_len_m,0,0],[0,axis_len_m,0],[0,0,axis_len_m]], dtype=np.float32)
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.astype(np.float32)
    pts2d, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    axes2d, _ = cv2.projectPoints(axes_pts, rvec, tvec, K, dist)
    pts2d = pts2d.squeeze().astype(int)
    axes2d = axes2d.squeeze().astype(int)
    # Draw points
    for (u,v) in pts2d:
        cv2.circle(img, (u,v), 5, point_color, -1)
    # Draw axes
    origin = tuple(axes2d[0])
    cv2.line(img, origin, tuple(axes2d[1]), (0,0,255), 2) # X: red
    cv2.line(img, origin, tuple(axes2d[2]), (0,255,0), 2) # Y: green
    cv2.line(img, origin, tuple(axes2d[3]), (255,0,0), 2) # Z: blue
    return img

def plot_3d_poses(Rt_list: list[tuple[np.ndarray,np.ndarray]], plane_extent=0.2) -> plt.Figure:
    """
    Plot world plane Z=0, axes, and camera frustums.
    Rt_list: list of (R, t)
    plane_extent: size of plane square
    Returns: matplotlib Figure
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot plane
    s = plane_extent/2
    plane = np.array([[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0],[ -s,-s,0]])
    ax.plot(plane[:,0], plane[:,1], plane[:,2], 'k-')
    # World axes
    ax.quiver(0,0,0,0.05,0,0,color='r')
    ax.quiver(0,0,0,0,0.05,0,color='g')
    ax.quiver(0,0,0,0,0,0.05,color='b')
    # Camera frustums
    for R, t in Rt_list:
        # Camera center
        C = t.flatten()
        # Frustum corners in camera frame
        frustum = np.array([[0,0,0],[0.01,0.01,-0.03],[-0.01,0.01,-0.03],[-0.01,-0.01,-0.03],[0.01,-0.01,-0.03],[0,0,0]])
        frustum_w = (R @ frustum.T).T + C
        ax.plot(frustum_w[:,0], frustum_w[:,1], frustum_w[:,2], 'b-')
        ax.scatter(C[0], C[1], C[2], c='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,0.7])
    ax.view_init(elev=30, azim=-60)
    return fig

# Smoke test
if __name__ == "__main__":
    img = np.zeros((480,640,3), dtype=np.uint8)
    K = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=np.float64)
    R = np.eye(3)
    t = np.array([[0],[0],[1]])
    XY = np.array([[0,0],[0.1,0],[0.1,0.1],[0,0.1]], dtype=np.float32)
    out_img = overlay_points_and_axes(img, K, None, R, t, XY)
    fig = plot_3d_poses([(R, t)])
    print("Smoke test passed.")
