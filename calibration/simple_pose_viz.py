#!/usr/bin/env python3
"""
Simple Camera Pose Visualization (Notebook Style)

This module provides simple functions to visualize camera poses in 3D space
following the same approach as the pinhole camera model notebook.
Based on the display_camera_poses() function from the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

try:
    import pytransform3d.transformations as pt
    import pytransform3d.camera as pc
    PYTRANSFORM3D_AVAILABLE = True
except ImportError:
    PYTRANSFORM3D_AVAILABLE = False
    print("Warning: pytransform3d not available. Install with: pip install pytransform3d")


def display_camera_poses(calibration_json_path: str, output_dir: str = None):
    """
    Display all camera poses in 3D following the notebook approach.
    
    Args:
        calibration_json_path: Path to calibration.json file
        output_dir: Optional directory to save the plot
    
    Returns:
        matplotlib Figure object
    """
    if not PYTRANSFORM3D_AVAILABLE:
        print("ERROR: pytransform3d not available. Install with: pip install pytransform3d")
        return None
        
    # Import here to avoid circular dependencies
    from .calib_io import load_calibration_json
    
    if not os.path.exists(calibration_json_path):
        print(f"ERROR: Calibration file not found: {calibration_json_path}")
        return None
        
    print("=== Displaying camera poses ===")
    
    # Load calibration data
    calibration = load_calibration_json(calibration_json_path)
    
    # Extract calibration parameters
    K = calibration["_K_np"]
    rvecs = [np.array(rv, dtype=np.float32).reshape(3, 1) for rv in calibration["rvecs"]]
    tvecs = [np.array(tv, dtype=np.float32).reshape(3, 1) for tv in calibration["tvecs"]]
    image_size = tuple(calibration["image_size"])
    
    print(f"Found {len(rvecs)} camera poses")
    print(f"Image size: {image_size}")
    print(f"RMS reprojection error: {calibration['rms']:.4f}")
    
    # Default image size for visualization (adjust as needed)
    nCols, nRows = image_size
    sensor_size = np.array([nCols, nRows])
    virtual_image_distance = 0.8
    
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Process each camera pose (following notebook approach)
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        print(f"Processing camera {i+1}/{len(rvecs)}")
        
        # Convert rotation vector to rotation matrix
        Omega, _ = cv2.Rodrigues(rvec)
        tau = tvec.reshape(3, 1)
        
        # Camera pose matrix [R^T | -R^T * t] for visualization
        # This converts from world-to-camera (calibration) to camera-to-world (visualization)
        Rt = np.block([Omega.T, -Omega.T @ tau])
        
        # Convert Rt from 3x4 to a 4x4 transformation matrix
        Rt = np.vstack([Rt, [0, 0, 0, 1]])
        
        print(f"Camera {i+1} pose matrix:\n{Rt}")
        
        # This is the camera coordinate frame
        cam2world = Rt
        
        # Plot camera coordinate frame (XYZ axes)
        # Scale axes to be visible relative to camera distances
        axis_scale = 200  # 200mm axes length - visible at camera distances
        pt.plot_transform(ax, A2B=cam2world, s=axis_scale, name=f"Cam_{i+1}")
        
        print(f"  -> Plotted XYZ axes for Camera {i+1} at position {cam2world[:3, 3]}")
        
        # Skip camera frustums - focus on coordinate axes for orientation
        # (Frustums can be enabled later if needed)
    
    # Plot the world coordinate frame (at origin) - represents the chessboard
    world_frame = np.eye(4)  # Identity matrix = origin
    print(f"World coordinate frame (chessboard):\n{world_frame}")
    # Make world axes larger and more prominent than camera axes
    world_axis_scale = 400  # 400mm axes length for world/chessboard frame
    pt.plot_transform(ax, A2B=world_frame, s=world_axis_scale, name="Chessboard")
    
    # Set plot properties
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Camera Poses in World Coordinates\nRMS Error: {calibration["rms"]:.4f}')
    
    # Calculate proper axis limits based on camera positions
    all_positions = []
    for rvec, tvec in zip(rvecs, tvecs):
        Omega, _ = cv2.Rodrigues(rvec)
        tau = tvec.reshape(3, 1)
        Rt = np.block([Omega.T, -Omega.T @ tau])
        camera_pos = Rt[:, 3]  # Extract camera position (translation part)
        all_positions.append(camera_pos)
    
    if all_positions:
        all_positions = np.array(all_positions)
        
        # Calculate reasonable bounds with some padding
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
        
        # Add 20% padding to bounds
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        padding = 0.2
        x_pad = x_range * padding
        y_pad = y_range * padding
        z_pad = z_range * padding
        
        # Include origin (world frame) in the view
        x_min = min(x_min - x_pad, -100)
        x_max = max(x_max + x_pad, 100)
        y_min = min(y_min - y_pad, -100)
        y_max = max(y_max + y_pad, 100)
        z_min = min(z_min - z_pad, -100)
        z_max = max(z_max + z_pad, 100)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        print(f"Camera positions range:")
        print(f"  X: {all_positions[:, 0].min():.1f} to {all_positions[:, 0].max():.1f}")
        print(f"  Y: {all_positions[:, 1].min():.1f} to {all_positions[:, 1].max():.1f}")
        print(f"  Z: {all_positions[:, 2].min():.1f} to {all_positions[:, 2].max():.1f}")
        print(f"Set axis limits:")
        print(f"  X: {x_min:.1f} to {x_max:.1f}")
        print(f"  Y: {y_min:.1f} to {y_max:.1f}")
        print(f"  Z: {z_min:.1f} to {z_max:.1f}")
    else:
        # Fallback limits
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)
    
    # Set a nice viewing angle
    ax.view_init(elev=30, azim=70)
    
    plt.tight_layout()
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "camera_poses_3d.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    return fig


def simple_pose_visualization(calibration_json_path: str, output_dir: str = None, show_plot: bool = False):
    """
    Main function for simple pose visualization.
    
    Args:
        calibration_json_path: Path to calibration.json file
        output_dir: Directory to save plots (optional)
        show_plot: Whether to display the plot interactively
    
    Returns:
        Dictionary with generated figures
    """
    print("=" * 60)
    print("SIMPLE CAMERA POSE VISUALIZATION")
    print("=" * 60)
    
    # Create the visualization
    fig = display_camera_poses(calibration_json_path, output_dir)
    
    if fig is None:
        print("FAILED: Could not create pose visualization")
        return {}
    
    # Show plot if requested
    if show_plot:
        print("Displaying plot... (close window to continue)")
        plt.show()
    
    print("SUCCESS: Camera pose visualization completed")
    return {"camera_poses_3d": fig}


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        calib_path = sys.argv[1]
    else:
        calib_path = "data/results/calibration.json"
    
    simple_pose_visualization(calib_path, "data/results/pose_viz", show_plot=True)
