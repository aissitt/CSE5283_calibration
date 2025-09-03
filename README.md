# CSE5283_calibration
Camera calibration core (Ari). OpenCV pipeline; UI and 3D vis come later.

## Setup (Conda)
```bash
conda env create -f environment.yml
conda activate calibration
```

Run
```bash
# Place .jpeg chessboard photos in data/images/
python scripts/run_calibration.py \
	--images_dir data/images \
	--cols 9 --rows 6 \
	--square_size 22.0 \
	--visualize_corners_to data/results/corners \
	--out_json data/results/calibration.json \
	--preview_image data/images/1.jpg \
	--preview_out data/results/undistort_preview.jpg \
	--undistort_alpha 0.85 \
	--undistort_center_pp
```

Outputs

data/results/calibration.json — intrinsics K, distortion, rvecs/tvecs, RMS + per-image RMSE.

data/results/corners/ — corner overlays for sanity checks.

data/results/undistort_preview.jpg — before/after undistortion.

Notes

Pattern size uses inner corners (OpenCV).

Square size is in millimeters.

calibration/utils.py mirrors key helpers from Prof. Ribeiro’s pinhole camera notes, so we can extend the same API later.


## Quick Sanity Check

```bash
conda activate calibration
python scripts/run_calibration.py --images_dir data/images --cols 9 --rows 6 --square_size 22.0 --visualize_corners_to data/results/corners --out_json data/results/calibration.json
```

Corner overlays appear in `data/results/corners/`. `calibration.json` contains K, distortion, rvecs/tvecs for Lamine’s pose vis and Blake's Gradio UI.
