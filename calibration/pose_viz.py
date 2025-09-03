"""
3D Camera Pose Visualization Module

This module provides functions to visualize camera poses in 3D space relative to
the world coordinate system (chessboard). It converts calibration results to 
camera-in-world poses and plots them using pytransform3d.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os

try:
    import pytransform3d.transformations as pt
    import pytransform3d.camera as pc
    PYTRANSFORM3D_AVAILABLE = True
except ImportError:
    PYTRANSFORM3D_AVAILABLE = False
    print("Warning: pytransform3d not available. Install with: pip install pytransform3d")


class CameraPoseVisualizer:
    """Stateless helper class for 3D camera pose visualization."""
    
    @staticmethod
    def rvec_tvec_to_camera_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Convert rotation vector and translation vector to 4x4 camera-to-world matrix.
        
        Args:
            rvec: Rotation vector (3x1) from cv2.calibrateCamera
            tvec: Translation vector (3x1) from cv2.calibrateCamera
            
        Returns:
            4x4 homogeneous transformation matrix (camera-to-world)
            
        Note:
            The calibration gives us world-to-camera transformation [R|t].
            For visualization, we need camera-to-world: [R^T | -R^T * t].
            This is because we want to show where the camera is located in world space.
        """
        import cv2
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Calibration gives us world->camera transformation [R|t]
        # We need camera->world transformation for plotting camera position
        # Camera-to-world: [R^T | -R^T * t]
        R_world_to_cam = R
        t_world_to_cam = tvec.reshape(3, 1)
        
        # Invert to get camera-to-world
        R_cam_to_world = R_world_to_cam.T
        t_cam_to_world = -R_cam_to_world @ t_world_to_cam
        
        # Build 4x4 homogeneous matrix
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = R_cam_to_world
        cam_to_world[:3, 3:4] = t_cam_to_world
        
        return cam_to_world
    
    @staticmethod
    def compute_camera_poses(rvecs: List[np.ndarray], tvecs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Convert calibration results to camera-in-world poses.
        
        Args:
            rvecs: List of rotation vectors from calibration
            tvecs: List of translation vectors from calibration
            
        Returns:
            List of 4x4 camera-to-world transformation matrices
            
        Note:
            Each matrix represents where one camera image was taken from,
            expressed in the world coordinate system (chessboard frame).
        """
        poses = []
        for rvec, tvec in zip(rvecs, tvecs):
            pose = CameraPoseVisualizer.rvec_tvec_to_camera_matrix(rvec, tvec)
            poses.append(pose)
        return poses
    
    @staticmethod
    def plot_camera_poses_3d(
        camera_poses: List[np.ndarray],
        K: np.ndarray,
        image_size: Tuple[int, int],
        title: str = "Camera Poses in World Coordinates",
        virtual_image_distance: float = 5.0,
        world_axes_scale: float = 10.0,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot 3D visualization of camera poses using pytransform3d.
        
        Args:
            camera_poses: List of 4x4 camera-to-world matrices
            K: Camera intrinsic matrix (3x3)
            image_size: (width, height) of images in pixels
            title: Plot title
            virtual_image_distance: Distance to draw virtual image planes
            world_axes_scale: Scale for world coordinate axes
            figsize: Figure size for matplotlib
            
        Returns:
            matplotlib Figure object
            
        Note:
            - World frame is at chessboard origin (corner)
            - X-axis: along chessboard rows (red)
            - Y-axis: along chessboard columns (green)  
            - Z-axis: perpendicular to chessboard (blue)
            - Cameras shown as colored frustrums with their viewing direction
        """
        if not PYTRANSFORM3D_AVAILABLE:
            raise ImportError("pytransform3d required for 3D visualization")
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot world coordinate frame at origin (chessboard)
        world_frame = np.eye(4)  # Identity = origin
        pt.plot_transform(ax, A2B=world_frame, s=world_axes_scale, name="World (Chessboard)")
        
        # Convert image size to sensor size for pytransform3d
        # (This is arbitrary scaling since we don't know actual sensor dimensions)
        sensor_size = np.array([image_size[0], image_size[1]]) / 100.0  # Scale down for viz
        
        # Plot each camera pose
        for i, cam_pose in enumerate(camera_poses):
            # Plot camera coordinate frame
            pt.plot_transform(ax, A2B=cam_pose, s=world_axes_scale * 0.5, 
                            name=f"Cam_{i+1:02d}")
            
            # Plot camera frustum (if pytransform3d camera module works)
            try:
                pc.plot_camera(
                    ax,
                    cam2world=cam_pose,
                    M=K,
                    sensor_size=sensor_size,
                    virtual_image_distance=virtual_image_distance,
                    alpha=0.3
                )
            except Exception as e:
                # Fallback: just plot the camera frame
                print(f"Note: Could not plot camera frustum for camera {i+1}: {e}")
        
        # Set equal aspect ratio and reasonable limits
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')  
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        
        # Compute reasonable axis limits based on camera positions
        all_positions = np.array([pose[:3, 3] for pose in camera_poses])
        if len(all_positions) > 0:
            center = np.mean(all_positions, axis=0)
            span = np.max(np.std(all_positions, axis=0)) * 3
            
            ax.set_xlim(center[0] - span, center[0] + span)
            ax.set_ylim(center[1] - span, center[1] + span)
            ax.set_zlim(center[2] - span, center[2] + span)
        else:
            # Default limits if no cameras
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.set_zlim(-50, 50)
        
        # Set a nice viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_camera_trajectory_2d(
        camera_poses: List[np.ndarray],
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot 2D top-down view of camera positions and orientations.
        
        Args:
            camera_poses: List of 4x4 camera-to-world matrices
            figsize: Figure size for matplotlib
            
        Returns:
            matplotlib Figure object
            
        Note:
            Shows bird's-eye view (X-Y plane) with camera positions as dots
            and orientation arrows showing where each camera was pointing.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        if len(camera_poses) == 0:
            ax1.text(0.5, 0.5, 'No camera poses to display', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No camera poses to display',
                    ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # Extract positions and orientations
        positions = np.array([pose[:3, 3] for pose in camera_poses])
        
        # Plot 1: Top-down view (X-Y plane)
        ax1.scatter(positions[:, 0], positions[:, 1], c='blue', s=50, alpha=0.7)
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3, linewidth=1)
        
        # Add arrows showing camera orientation (where it's looking)
        for i, pose in enumerate(camera_poses):
            pos = pose[:3, 3]
            # Camera Z-axis (pointing direction) in world coords
            cam_z = pose[:3, 2]  # 3rd column of rotation matrix
            # Draw arrow showing viewing direction
            arrow_scale = 5.0
            ax1.arrow(pos[0], pos[1], cam_z[0] * arrow_scale, cam_z[1] * arrow_scale,
                     head_width=1.0, head_length=1.5, fc='red', ec='red', alpha=0.7)
            ax1.text(pos[0], pos[1], f'{i+1}', fontsize=8, ha='center', va='center')
        
        ax1.scatter([0], [0], c='red', marker='s', s=100, label='World Origin')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title('Camera Positions (Top View)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # Plot 2: Side view (X-Z plane)
        ax2.scatter(positions[:, 0], positions[:, 2], c='blue', s=50, alpha=0.7)
        ax2.plot(positions[:, 0], positions[:, 2], 'b-', alpha=0.3, linewidth=1)
        
        for i, pose in enumerate(camera_poses):
            pos = pose[:3, 3]
            ax2.text(pos[0], pos[2], f'{i+1}', fontsize=8, ha='center', va='center')
        
        ax2.scatter([0], [0], c='red', marker='s', s=100, label='World Origin')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Z (mm)')
        ax2.set_title('Camera Positions (Side View)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig


def visualize_calibration_poses(
    calibration_json_path: str,
    output_dir: str = None,
    show_3d: bool = True,
    show_2d: bool = True
) -> Dict[str, plt.Figure]:
    """
    Complete pipeline: load calibration and create pose visualizations.
    
    Args:
        calibration_json_path: Path to calibration.json file
        output_dir: If provided, save plots to this directory
        show_3d: Whether to create 3D pose visualization
        show_2d: Whether to create 2D trajectory plots
        
    Returns:
        Dictionary mapping plot names to matplotlib Figure objects
        
    Note:
        This is the main function that demonstrates the complete workflow:
        1. Load calibration results
        2. Convert poses to camera-in-world coordinates
        3. Create 3D and/or 2D visualizations
        4. Optionally save plots
    """
    # Import here to avoid circular dependencies
    from .calib_io import load_calibration_json
    
    # Load calibration data
    print(f"Loading calibration from: {calibration_json_path}")
    calib = load_calibration_json(calibration_json_path)
    
    # Extract parameters
    K = calib["_K_np"]
    rvecs = [np.array(rv, dtype=np.float32).reshape(3, 1) for rv in calib["rvecs"]]
    tvecs = [np.array(tv, dtype=np.float32).reshape(3, 1) for tv in calib["tvecs"]]
    image_size = tuple(calib["image_size"])
    
    print(f"Found {len(rvecs)} camera poses")
    print(f"Image size: {image_size}")
    print(f"RMS reprojection error: {calib['rms']:.4f}")
    
    # Convert to camera-in-world poses
    camera_poses = CameraPoseVisualizer.compute_camera_poses(rvecs, tvecs)
    
    figures = {}
    
    # Create 3D visualization
    if show_3d:
        print("Creating 3D pose visualization...")
        try:
            fig_3d = CameraPoseVisualizer.plot_camera_poses_3d(
                camera_poses, K, image_size,
                title=f"Camera Poses - RMS Error: {calib['rms']:.4f}"
            )
            figures['3d_poses'] = fig_3d
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, 'camera_poses_3d.png')
                fig_3d.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved 3D plot: {output_path}")
                
        except Exception as e:
            print(f"Failed to create 3D visualization: {e}")
    
    # Create 2D trajectory plots
    if show_2d:
        print("Creating 2D trajectory plots...")
        try:
            fig_2d = CameraPoseVisualizer.plot_camera_trajectory_2d(camera_poses)
            figures['2d_trajectory'] = fig_2d
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, 'camera_trajectory_2d.png')
                fig_2d.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved 2D plot: {output_path}")
                
        except Exception as e:
            print(f"Failed to create 2D visualization: {e}")
    
    return figures
