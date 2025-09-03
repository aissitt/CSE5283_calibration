# 10-Minute Class Presentation Script
**CSE5283 Camera Calibration Assignment Demo**

---

## Preparation (Before Class)
- [ ] Laptop charged, backup power adapter
- [ ] Test internet connection for Colab
- [ ] Have backup: screenshots of key results 
- [ ] Print chessboard pattern (if demonstrating live capture)
- [ ] USB drive with slides/notebook file (classroom backup)

---

## Demo Flow (10 minutes total)

### Slide 1: Title & Overview (1 minute)
**"Camera Calibration with OpenCV and Gradio UI"**

- **What we built**: Complete calibration pipeline with interactive UI
- **Key deliverables**: 
  - 20 chessboard images captured
  - OpenCV calibration pipeline  
  - Gradio interface for easy use
  - 3D camera pose visualization
  - Axes overlay verification

---

### Demo Part 1: Quick Calibration Results (2 minutes)

**Show completed calibration output**
- Open saved `calibration.json` - show K matrix, distortion coefficients
- **Key numbers to highlight**:
  - RMS reprojection error: X.XX pixels
  - Number of images used: XX/20
  - Focal length fx, fy: XXX, XXX pixels
  - Principal point: (XXX, XXX)

**"This tells us our camera's internal geometry - how it maps 3D world to 2D pixels"**

---

### Demo Part 2: Axes Overlay Verification (2 minutes)

**Show `data/results/axes/` images**
- Display 3-4 sample images with coordinate axes overlaid
- **Point out**:
  - Red line = X-axis (along chessboard rows)
  - Green line = Y-axis (along chessboard columns)  
  - Blue line = Z-axis (perpendicular to board)
  - Black dot = World origin (chessboard corner)

**"These overlays prove our calibration is working - the 3D world frame projects correctly onto each image"**

---

### Demo Part 3: 3D Camera Pose Visualization (2 minutes)

**Show `data/results/pose_viz/camera_poses_3d.png`**
- **Explain the plot**:
  - Red/green/blue frame at origin = World (chessboard)
  - Colored camera frames = where each photo was taken from
  - Camera frustrums show viewing direction
  
**Show `camera_trajectory_2d.png`**
- Top view and side view of camera movement
- "This shows I moved around the chessboard systematically"

**"This visualization proves we recovered the full 3D geometry - where the camera was for each photo"**

---

### Demo Part 4: Live Gradio Interface (2.5 minutes)

**Open Colab notebook**
- Run first cell (install packages) - while it installs, explain:
  - "Self-contained notebook, works in any browser"
  - "All dependencies installed with pip"
  - "Uses our modular code structure"

**Show Gradio UI components**:
- File upload area
- Parameter inputs (corners, square size)
- "Run Calibration" button
- Visualization outputs

**Quick run through a mini-calibration** (if time permits):
- Upload 3-4 images
- Set parameters: 9x6 corners, 25mm squares
- Click "Run Calibration"
- Show results appear in tabs

**"The UI makes it easy for anyone to calibrate their camera - no command line needed"**

---

### Demo Part 5: Technical Understanding (0.5 minutes)

**Quick math explanation**:
- "The pinhole model: 3D world point → camera coordinates → image pixels"
- "Key equation: x = K[R|t]X, where K=intrinsics, [R|t]=camera pose"
- "Calibration finds K from multiple views of known 3D pattern"

---

## Backup Slides (If Demo Fails)

### Backup: Screenshots
- Have static images of all key results
- Calibration JSON content
- Sample axes overlays
- 3D pose plots
- Gradio interface

### Backup: Architecture
- Code structure diagram
- Show modular design: calibration/, scripts/, notebooks/
- Explain stateless helper classes

---

## Q&A Preparation

### Likely Questions:
1. **"How accurate is your calibration?"**
   - RMS error of X.XX pixels
   - Validated with axes overlays
   - Consistent across XX/20 images

2. **"Why use Gradio instead of just command line?"**
   - Accessibility for non-programmers
   - Visual feedback
   - Integrated workflow

3. **"What was hardest part?"**
   - Coordinate system conversions (camera↔world)
   - Ensuring proper [R^T | -R^T*t] for visualization
   - UI integration with existing codebase

4. **"How did you validate results?"**
   - Visual inspection of axes overlays
   - Reprojection error analysis
   - 3D pose consistency checks

---

## Timing Checkpoints
- **2 min**: Finished showing calibration results
- **4 min**: Finished axes overlay explanation  
- **6 min**: Finished 3D pose visualization
- **8.5 min**: Finished Gradio demo
- **9 min**: Finished technical summary
- **10 min**: Open for questions

---

## Key Takeaways to Emphasize
1. **Complete pipeline**: Data → Calibration → Validation → Visualization
2. **Practical tool**: Anyone can use the Gradio interface
3. **Mathematical understanding**: Know what the matrices mean
4. **Visual validation**: Multiple ways to verify calibration quality
