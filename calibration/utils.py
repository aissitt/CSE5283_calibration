import os, glob, json
import cv2
import numpy as np

# io helpers
def sorted_image_paths(folder, patterns=(".jpg", ".jpeg", ".JPG", ".JPEG")):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(folder, f"*{p}")))
    # On case-insensitive filesystems multiple patterns may match the same file
    # (e.g. '*.jpg' and '*.JPG'). Deduplicate while preserving order.
    seen = set()
    unique = []
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return sorted(unique)

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

# math helpers
class Utils:
    @staticmethod
    def project_points(W: np.ndarray, Lambda: np.ndarray, Rt: np.ndarray) -> np.ndarray:
        """
        Projects 3D points W (3xN) to image plane using Lambda (3x3) and Rt (3x4).
        """
        W_tilde = np.vstack((W, np.ones((1, W.shape[1]))))
        print(f"W_tilde = \n{W_tilde}\n")
        X_tilde = Lambda @ Rt @ W_tilde
        print(f"X_tilde = \n{X_tilde}\n")
        X = X_tilde[:2, :] / X_tilde[2:3, :]
        print(f"Projected image points (inhom.) = \n{X}\n")
        return X

    @staticmethod
    def draw_coordinate_frame(image_points_2xN: np.ndarray, img_bgr: np.ndarray, thickness: int = 2):
        """
        Draw simple RGB axes using already-projected [origin, X, Y, Z] points (2x4).
        """
        img = img_bgr.copy()
        pts = image_points_2xN.T.astype(int)
        origin = tuple(pts[0])
        cv2.line(img, origin, tuple(pts[1]), (0, 0, 255), thickness)  # X (red)
        cv2.line(img, origin, tuple(pts[2]), (0, 255, 0), thickness)  # Y (green)
        cv2.line(img, origin, tuple(pts[3]), (255, 0, 0), thickness)  # Z (blue)
        return img

    @staticmethod
    def build_Lambda(phi_x: float, phi_y: float, delta_x: float, delta_y: float) -> np.ndarray:
        """
        Intrinsic matrix layout from the notes:
        [phi_x   0    delta_x]
        [  0   phi_y  delta_y]
        [  0     0        1  ]
        """
        return np.array([[phi_x, 0.0,   delta_x],
                         [0.0,   phi_y, delta_y],
                         [0.0,   0.0,   1.0   ]], dtype=float)

    @staticmethod
    def json_read(filename: str):
        with open(os.path.abspath(filename)) as f:
            return json.load(f)
