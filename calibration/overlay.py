"""
Overlay functionality for drawing world coordinate axes on calibrated images.

This module provides functions to project and draw 3D world coordinate frames
onto images using camera calibration parameters (K, dist, rvec, tvec).
"""

import os
import numpy as np
import cv2
from typing import Tuple, List


class AxesOverlay:
    """Stateless helper class for drawing world coordinate axes on images."""
    
    @staticmethod
    def create_world_axes(axis_length: float = 3.0, origin_world: Tuple[float, float, float] | None = None) -> np.ndarray:
        """
        Create 3D points representing world coordinate axes.
        
        Args:
            axis_length: Length of each axis in world units (same as square_size)
            origin_world: Optional origin (u0, v0, w0) in world units. If None, uses (0,0,0)
            
        Returns:
            World points (4x3): origin, X-axis end, Y-axis end, Z-axis end
            
        Note:
            - Origin is at (0,0,0) - the corner of the chessboard
            - X-axis points along chessboard rows (red when drawn)
            - Y-axis points along chessboard columns (green when drawn)  
            - Z-axis points up from the chessboard plane (blue when drawn)
            
        Reference implementation from pinhole_camera_model.py:
            W = scale_factor * np.array([
                [ 0, 1,  0,  0],  # [origin, X-end, Y-end, Z-end] as columns
                [ 0, 0,  1,  0],
                [ 0, 0,  0,  1]
            ])
        """
        if origin_world is None:
            origin_world = (0.0, 0.0, 0.0)
        u0, v0, w0 = origin_world
        
        # Create coordinate frame points following reference implementation
        # Note: Reference uses columns for points, we use rows for cv2.projectPoints
        return np.array([
            [u0,            v0,            w0],            # Origin
            [u0+axis_length, v0,            w0],            # X-axis (red)
            [u0,            v0+axis_length, w0],            # Y-axis (green)
            [u0,            v0,            w0+axis_length]  # Z-axis (blue, positive Z - FIXED!)
        ], dtype=np.float32)
    
    @staticmethod
    def project_axes_to_image(
        world_axes: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray
    ) -> np.ndarray:
        """
        Project 3D world axes points to 2D image coordinates.
        
        Args:
            world_axes: 3D points (4x3) representing origin and axis endpoints
            rvec: Rotation vector (3x1) from cv2.calibrateCamera
            tvec: Translation vector (3x1) from cv2.calibrateCamera  
            K: Camera intrinsic matrix (3x3)
            dist: Distortion coefficients (5x1 or 8x1)
            
        Returns:
            Image points (4x2): projected coordinates of axes in pixels
            
        Note:
            Uses cv2.projectPoints which handles the full pinhole camera model:
            x_image = K * [R|t] * X_world, then perspective division and undistortion
            
        Reference implementation shows correct format:
            image_axes, jac = cv2.projectPoints(W, rvec, tvec, Lambda, distCoeffs)
            image_axes = image_axes.squeeze().T  # Remove extra brackets and transpose
        """
        # Ensure proper input format for cv2.projectPoints
        # world_axes should be (N, 3) where N=4 for [origin, X, Y, Z]
        world_points = world_axes.astype(np.float32)
        
        # Ensure rvec and tvec have correct shapes for cv2.projectPoints
        rvec_input = rvec.reshape(3, 1) if rvec.shape != (3, 1) else rvec
        tvec_input = tvec.reshape(3, 1) if tvec.shape != (3, 1) else tvec
        
        # Project 3D world points to 2D image points
        image_points, _ = cv2.projectPoints(
            world_points,    # 3D object points (N, 3)
            rvec_input,     # Rotation vector (3, 1)
            tvec_input,     # Translation vector (3, 1)
            K,              # Camera matrix (3, 3)
            dist            # Distortion coefficients
        )
        
        # cv2.projectPoints returns shape (N, 1, 2), flatten to (N, 2)
        # Following reference: image_axes.squeeze().T for shape (2, N)
        # But we'll return (N, 2) and handle the transpose in drawing function
        return image_points.reshape(-1, 2)
    
    @staticmethod
    def draw_axes_on_image(
        img: np.ndarray,
        image_axes: np.ndarray,
        thickness: int | None = None,
        axis_colors: Tuple[Tuple[int, int, int], ...] = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    ) -> np.ndarray:
        """
        Draw coordinate axes on an image.
        
        Args:
            img: Input image (BGR format)
            image_axes: Projected axis points (4x2) - origin + 3 axis endpoints
            thickness: Line thickness for drawing axes
            axis_colors: Colors for X, Y, Z axes in BGR format
                        Default: X=red, Y=green, Z=blue
                        
        Returns:
            Image with axes drawn (copy of input)
            
        Note:
            - Origin gets a black circle for visibility
            - Axes are drawn as arrows from origin to endpoints
            - Colors follow computer vision convention: X=red, Y=green, Z=blue
        """
        result = img.copy()
        # Auto thickness based on image size if not provided - improved for better visibility
        if thickness is None:
            h, w = result.shape[:2]
            # Use larger thickness for better visibility, matching reference (thickness=5)
            thickness = max(5, int(min(h, w) / 200))
        
        # Convert to integer coordinates for drawing
        # image_axes is (4, 2) - need to transpose for reference compatibility
        points = image_axes.astype(int)
        
        # Reference implementation: x0, y0 = image_points[:,0].astype(int)
        # This means image_points is (2, 4) where columns are points
        # Since we have (4, 2), we need to transpose
        image_points_ref_format = points.T  # Now (2, 4)
        
        # Extract coordinates following reference implementation exactly
        x0, y0 = image_points_ref_format[:, 0].astype(int)  # Origin
        origin = (x0, y0)
        
        # Draw origin as a black circle - match reference (radius=9)
        cv2.circle(result, origin, 9, (0, 0, 0), -1)
        
        # Draw axes following reference implementation exactly
        # X-axis: RED (255, 0, 0) in BGR
        x1, y1 = image_points_ref_format[:, 1].astype(int)
        cv2.arrowedLine(result, origin, (x1, y1), (0, 0, 255), thickness)  # X=red
        
        # Y-axis: GREEN (0, 255, 0) in BGR  
        x2, y2 = image_points_ref_format[:, 2].astype(int)
        cv2.arrowedLine(result, origin, (x2, y2), (0, 255, 0), thickness)  # Y=green
        
        # Z-axis: BLUE (0, 0, 255) in BGR
        x3, y3 = image_points_ref_format[:, 3].astype(int)
        cv2.arrowedLine(result, origin, (x3, y3), (255, 0, 0), thickness)  # Z=blue
        
        return result

    @staticmethod
    def draw_axes_2d_at_image_center(
        img: np.ndarray,
        length_px: int = 100,
        thickness: int | None = None,
        axis_colors: Tuple[Tuple[int, int, int], ...] = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    ) -> np.ndarray:
        """
        Draw 2D axes at the image center for debugging/visualization.
        X axis points right (red), Y axis points down (green), Z axis drawn as
        a short arrow up-left (blue) to suggest out-of-plane.
        """
        h, w = img.shape[:2]
        cx, cy = int(w/2), int(h/2)
        if thickness is None:
            # Use same thickness calculation as projected axes for consistency
            thickness = max(4, int(min(h, w) / 200))

        result = img.copy()
        origin = (cx, cy)
        
        # Draw origin as a black circle - same size as projected axes
        cv2.circle(result, origin, radius=max(8, thickness*3), color=(0, 0, 0), thickness=-1)
        
        # X to the right (red)
        cv2.arrowedLine(result, origin, (cx + length_px, cy), axis_colors[0], thickness, tipLength=0.5)
        # Y down (green)
        cv2.arrowedLine(result, origin, (cx, cy + length_px), axis_colors[1], thickness, tipLength=0.5)
        # Z pseudo (up-left, blue)
        cv2.arrowedLine(result, origin, (cx - int(0.7*length_px), cy - int(0.7*length_px)), axis_colors[2], thickness, tipLength=0.5)
        
        return result
    
    @staticmethod
    def overlay_axes_on_image(
        img_path: str,
        rvec: np.ndarray,
        tvec: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        axis_length: float = 3.0,
        origin_world: Tuple[float, float, float] | None = None,
        thickness: int | None = None,
        output_path: str = None
    ) -> np.ndarray:
        """
        Complete pipeline: load image, project axes, draw, optionally save.
        
        Args:
            img_path: Path to input image
            rvec: Rotation vector for this image
            tvec: Translation vector for this image
            K: Camera intrinsic matrix
            dist: Distortion coefficients
            axis_length: Length of coordinate axes in world units
            output_path: If provided, save result to this path
            
        Returns:
            Image with coordinate axes overlaid
            
        Note:
            This is the main function that combines all steps:
            1. Load image
            2. Create 3D world axes
            3. Project to image coordinates  
            4. Draw axes on image
            5. Optionally save result
        """
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        
        # Create 3D world coordinate axes at desired origin
        world_axes = AxesOverlay.create_world_axes(axis_length, origin_world)
        
        # Project 3D axes to 2D image coordinates
        try:
            image_axes = AxesOverlay.project_axes_to_image(world_axes, rvec, tvec, K, dist)
            
            # Check if projected axes are within image bounds and visible
            h, w = img.shape[:2]
            axes_visible = True
            
            # Check if origin is within reasonable bounds (not too far outside)
            origin_x, origin_y = image_axes[0]
            margin = min(w, h) * 0.2  # Allow 20% margin outside image
            
            if (origin_x < -margin or origin_x > w + margin or 
                origin_y < -margin or origin_y > h + margin):
                axes_visible = False
            else:
                # Check if axes are too small (less than 10 pixels)
                origin = image_axes[0]
                for i in range(1, len(image_axes)):
                    endpoint = image_axes[i]
                    dist_px = np.sqrt((endpoint[0] - origin[0])**2 + (endpoint[1] - origin[1])**2)
                    if dist_px < 10:  # Too small to see
                        axes_visible = False
                        break
            
            if axes_visible:
                # Draw projected axes
                result = AxesOverlay.draw_axes_on_image(img, image_axes, thickness=thickness)
            else:
                # Fallback: draw axes at image center
                print(f"Warning: Projected axes not visible in {os.path.basename(img_path)}, using center fallback")
                length_px = int(axis_length if axis_length is not None else 100)
                result = AxesOverlay.draw_axes_2d_at_image_center(img, length_px, thickness)
                
        except Exception as e:
            # Fallback: draw axes at image center if projection fails
            print(f"Warning: Projection failed for {os.path.basename(img_path)}: {e}, using center fallback")
            length_px = int(axis_length if axis_length is not None else 100)
            result = AxesOverlay.draw_axes_2d_at_image_center(img, length_px, thickness)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, result)
            
        return result


def overlay_axes_on_calibration_images(
    calibration_json_path: str,
    images_folder: str,
    output_folder: str,
    max_images: int = 10,
    axis_length: float | None = None,
    origin_mode: str = "corner",
    thickness: int | None = None
) -> List[str]:
    """
    Batch process: overlay axes on multiple calibrated images.
    
    Args:
        calibration_json_path: Path to calibration.json with K, dist, rvecs, tvecs
        images_folder: Folder containing original images  
        output_folder: Where to save images with axes overlaid
        max_images: Maximum number of images to process
        axis_length: Length of coordinate axes in world units
        
    Returns:
        List of output file paths created
        
    Note:
        This function demonstrates the practical use case:
        1. Load calibration results from JSON
        2. Process subset of images that had successful corner detection
        3. Save results for visual verification of calibration quality
    """
    # Import here to avoid circular dependencies
    from .calib_io import load_calibration_json
    from .utils import sorted_image_paths
    from .chessboard import find_corners_in_folder
    
    # Load calibration data
    calib = load_calibration_json(calibration_json_path)
    K = calib["_K_np"]
    dist = calib["_dist_np"]
    rvecs = [np.array(rv, dtype=np.float32).reshape(3, 1) for rv in calib["rvecs"]]
    tvecs = [np.array(tv, dtype=np.float32).reshape(3, 1) for tv in calib["tvecs"]]
    # Use a visually meaningful default axis length: 10 squares of the chessboard
    if axis_length is None or axis_length <= 0:
        cols, rows = (calib.get("pattern_size", [9, 6]) or [9, 6])
        # default square size (mm)
        square = float(calib.get("square_size", 22.0))
        # Use a larger scale factor for better visibility, similar to reference notebook
        # Scale factor of 10 makes axes clearly visible
        scale_factor = 10.0
        axis_length = float(scale_factor * square)
    
    # Try to recover the exact list/order of images that produced rvecs/tvecs
    # by re-detecting corners. This helps ensure pose-image pairing is correct
    # even if some images lacked corners.
    try:
        pattern_size = tuple(int(v) for v in calib.get("pattern_size", []))
        if len(pattern_size) == 2:
            _imgpoints, _image_size, used_paths = find_corners_in_folder(
                images_folder, pattern_size, visualize_to=None
            )
            image_paths = used_paths
        else:
            image_paths = sorted_image_paths(images_folder)
    except Exception:
        image_paths = sorted_image_paths(images_folder)

    # Compute origin in world coordinates (corner or center of board)
    origin_world = (0.0, 0.0, 0.0)
    if len(pattern_size) == 2:
        cols, rows = pattern_size
        square = float(calib.get("square_size", 22.0))
        if origin_mode.lower() == "center":
            origin_world = (
                (cols-1) * square * 0.5,
                (rows-1) * square * 0.5,
                0.0
            )
        elif origin_mode.lower() == "board_center":
            # Place origin at the center of the chessboard for better visibility
            origin_world = (
                (cols-1) * square * 0.5,
                (rows-1) * square * 0.5,
                0.0
            )
    
    # Process up to max_images
    output_paths = []
    processed = 0
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Warn if counts do not match
    if len(image_paths) != len(rvecs):
        print(
            f"Warning: number of poses ({len(rvecs)}) does not match number of selected images ({len(image_paths)})."
        )

    if origin_mode.lower() in ["image_center", "always_center"]:
        # Draw on image center for up to max_images, ignore rvec/tvec
        for img_path in image_paths[:max_images]:
            try:
                basename = os.path.basename(img_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(output_folder, f"{name}_axes{ext}")
                img = cv2.imread(img_path)
                length_px = int(axis_length if axis_length is not None else 100)
                result = AxesOverlay.draw_axes_2d_at_image_center(img, length_px, thickness)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, result)
                output_paths.append(output_path)
                processed += 1
                print(f"Processed {basename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"Failed to process {os.path.basename(img_path)}: {e}")
                continue
    else:
        for i, (img_path, rvec, tvec) in enumerate(zip(image_paths, rvecs, tvecs)):
            if processed >= max_images:
                break
            try:
                # Generate output filename
                basename = os.path.basename(img_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(output_folder, f"{name}_axes{ext}")
                # Overlay axes and save
                AxesOverlay.overlay_axes_on_image(
                    img_path, rvec, tvec, K, dist, axis_length,
                    origin_world=origin_world, thickness=thickness, output_path=output_path
                )
                output_paths.append(output_path)
                processed += 1
                print(f"Processed {basename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"Failed to process {os.path.basename(img_path)}: {e}")
                continue
    
    print(f"Successfully created {len(output_paths)} images with axes overlays")
    return output_paths
