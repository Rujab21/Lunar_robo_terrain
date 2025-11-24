# Lunar Robo Terrain Perception

This project implements a **hybrid 3D terrain perception system** for lunar robotics, combining **stereo vision depth estimation** with **object segmentation** using YOLOv8 and SAM. It detects craters and rocks, calculates their depths, generates occupancy grids, and creates 3D point cloud visualizations.

---

## Features

1. **Stereo Vision Depth Estimation**
   - Uses rectified left and right stereo images to compute disparity maps.
   - Applies **Semi-Global Block Matching (SGBM)** and **WLS filtering** for high-quality depth maps.
   - Depth map conversion to meters using camera intrinsic parameters.

2. **Object Segmentation**
   - **YOLOv8** model trained for crater detection.
   - **SAM (Segment Anything Model)** masks for rock segmentation.
   - Hybrid segmentation allows differentiation between craters and rocks in both images and point clouds.

3. **Depth Calculation**
   - Computes **minimum, mean, median, and maximum depth** for all detected objects.
   - Handles invalid or missing depth values gracefully.

4. **Overlay Visualization**
   - Clean and detailed overlays on original images:
     - **Clean overlay**: mask contours + depth labels.
     - **Detailed overlay**: contours, masks, bounding boxes, and depth text.
   - Visualizes detected objects with class-specific colors:
     - Craters → Yellow
     - Rocks → Blue/Red

5. **3D Point Cloud Visualization**
   - Generates a **colored 3D point cloud** using Open3D.
   - Applies hybrid segmentation colors:
     - Craters → Yellow
     - Rocks → Green
     - Background → Original colors
   - Supports interactive 3D visualization.

6. **Occupancy Grid Creation**
   - Generates **shape-preserving occupancy grids** for navigation.
   - Supports:
     - Binary occupancy grids
     - Color-coded grids preserving object shapes
     - Grid resolution customization
   - Flipped visualization for robot-centric perspective.

7. **Enhanced Terrain Analysis**
   - Provides **histograms and statistics** for object depth distributions.
   - Highlights **object distribution**, depth ranges, and robot Field of View (FOV).

---

## Dependencies

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- PIL (Pillow)
- Open3D
- Pandas
- glob
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- SAM masks (saved in `sam_masks/objects` folder)

---

## Usage

1. **Prepare stereo images:**
   - Place left and right images as `left.jpg` and `right.jpg`.

2. **Train or load YOLOv8 crater model:**
   trained yolov8 segmentation model link is :https://drive.google.com/drive/folders/1IcObGvpFIN4PBNauAK999OwFoB-z6T2q?usp=sharing
   ```python
   from ultralytics import YOLO
   crater_model = YOLO('best_crater_model.pt')

4. **Place SAM rock masks:**

    Masks should be in sam_masks/objects/mask_*.png.

    Rocks should have "class1" in their filename.

5. **Run the script:**

    python lunar_robo_terrain.py


## Outputs:

    Depth overlays: depths/object_detection_overlay_clean.png, depths/object_detection_overlay_detailed.png.

    3D point cloud: depths/point_cloud_hybrid_segmentation.ply.

    Occupancy grids and statistics displayed via matplotlib.

## Folder Structure
        Lunar_Robo_Terrain/
        │
        ├─ best_crater_model.pt          # Trained YOLOv8 crater model
        ├─ left.jpg                      # Left stereo image
        ├─ right.jpg                     # Right stereo image
        ├─ sam_masks/objects/            # SAM masks for rocks
        │    └─ mask_*.png
        ├─ depths/                       # Output folder
        │    ├─ object_detection_overlay_clean.png
        │    ├─ object_detection_overlay_detailed.png
        │    └─ point_cloud_hybrid_segmentation.ply
        └─ lunar_robo_terrain.py         # Main script

## Visualization

    Disparity Map

    Normalized Depth Map

    Real Depth Map

    3D Point Cloud

    Occupancy Grid

    Depth Histograms

    Colors in overlays and point clouds:

    Craters → Yellow

    Rocks → Red / Blue / Green depending on visualization

    Background → Original RGB

## Notes

    Make sure stereo images are rectified for accurate depth calculation.

    Adjust camera parameters (fx, fy, baseline_m) to match your setup.

    WLS filtering and median blur improve depth map quality.

    The occupancy grid resolution can be customized for finer navigation maps.
