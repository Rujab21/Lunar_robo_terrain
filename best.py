import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
import glob
from PIL import Image
import pandas as pd
from ultralytics import YOLO

# ===========================================================
# === LOAD YOUR TRAINED YOLOv8 CRATER MODEL ===
# ===========================================================
print("Loading YOLOv8 crater segmentation model...")
crater_model = YOLO('best_crater_model.pt')  # Your trained model path

# ===========================================================
# === STEREO VISION PIPELINE (UNCHANGED) ===
# ===========================================================

# --- Load stereo images with error handling ---
left = cv2.imread('left.jpg')
right = cv2.imread('right.jpg')

# Check if images loaded properly
if left is None or right is None:
    raise FileNotFoundError("Could not load one or both images. Check file paths and names.")

# Verify image sizes match
if left.shape != right.shape:
    raise ValueError("Left and right images must have the same dimensions")

# --- Convert to grayscale ---
grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

# Apply CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
grayL = clahe.apply(grayL)
grayR = clahe.apply(grayR)

# --- Create SGBM matcher with optimized parameters ---
window_size = 5
min_disp = 0
# Ensure num_disp is divisible by 16 and reasonable for your scene
num_disp = 16 * 15 # Reduced from 336 to 160 for better performance

left_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=150,
    uniquenessRatio=10,  # Increased from 0 to reduce false matches
    speckleWindowSize=0,
    speckleRange=32,
    preFilterCap=63,     # Added for better pre-filtering
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# --- Right matcher (for WLS) ---
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# --- Compute disparities ---
print("Computing disparities...")
disp_left = left_matcher.compute(grayL, grayR).astype(np.float32)
disp_right = right_matcher.compute(grayR, grayL).astype(np.float32)

# --- WLS filter setup ---
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000.0)
wls_filter.setSigmaColor(1.5)

# --- Apply WLS filtering ---
print("Applying WLS filter...")
filtered_disp = wls_filter.filter(disp_left, left, disparity_map_right=disp_right)
# Add median filter to reduce noise and fill small holes
filtered_disp = cv2.medianBlur(filtered_disp.astype(np.uint16), 5)  # Kernel size 5-7; convert to uint16 for blur

# --- Convert disparity to float properly ---
disp_float = filtered_disp.astype(np.float32) / 16.0

# Create mask for invalid disparities
disp_mask = (disp_float > 0) & (disp_float < num_disp)
disp_float[~disp_mask] = np.nan

# --- Debug disparity values ---
valid_disp = disp_float[disp_mask]
print(f"Disparity range: {np.nanmin(valid_disp):.2f} to {np.nanmax(valid_disp):.2f}")
print(f"Valid disparity pixels: {valid_disp.size}")
print(f"Mean disparity: {np.nanmean(valid_disp):.2f}")
print(f"Median disparity: {np.nanmedian(valid_disp):.2f}")

# === CAMERA PARAMETERS ===
baseline_m = 0.25  # meters
fx = 834.06        # focal length in pixels

# --- Compute depth map (Z = f * B / disparity) ---
depth_map = (baseline_m * fx) / disp_float

# --- Report valid depth range ---
valid_depth = depth_map[disp_mask]
print(f"Depth range (m): {np.nanmin(valid_depth):.2f} to {np.nanmax(valid_depth):.2f}")

# --- Adaptive visualization range ---
depth_clip_max = min(5, np.nanpercentile(valid_depth, 99))
print(f"Using display clip range 0–{depth_clip_max:.1f} m")

depth_vis = np.clip(depth_map, 0, depth_clip_max)
depth_vis = np.nan_to_num(depth_vis, nan=0.0)
depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
depth_vis = np.uint8(depth_vis)

# --- Visualization ---
# --- Visualization ---
plt.figure(figsize=(14, 10))

# Disparity map
plt.subplot(2, 2, 1)
plt.imshow(disp_float, cmap='jet',
           vmin=np.nanpercentile(valid_disp, 5),
           vmax=np.nanpercentile(valid_disp, 95))
plt.title('WLS Filtered Disparity')
plt.colorbar(label='Disparity (px)')

# Normalized Depth map (for visual clarity)
plt.subplot(2, 2, 2)
plt.imshow(depth_vis, cmap='jet')
plt.title(f'Depth Map (0–{depth_clip_max:.1f} m, normalized)')
plt.colorbar(label='Depth (normalized)')

# Real Depth map in meters
plt.subplot(2, 2, 3)
plt.imshow(depth_map, cmap='jet')
plt.title("Real Depth Map (meters)")
plt.colorbar(label='Depth (m)')

# Depth histogram
plt.subplot(2, 2, 4)
plt.hist(valid_depth.flatten(), bins=200, color='orange')
plt.title("Depth Distribution")
plt.xlabel("Depth (m)")
plt.ylabel("Pixel Count")

plt.tight_layout()
plt.show()

# ===========================================================
# === CALCULATE DEPTHS FOR ALL OBJECTS ===
# ===========================================================
print("\nCalculating depths for all objects...")

# Store depth results for all objects
depth_results = []

# ===========================================================
# === STEP 1: YOLOv8 CRATER DEPTH CALCULATION ===
# ===========================================================
print("Processing YOLOv8 craters...")
yolo_results = crater_model.predict(left, conf=0.25, imgsz=640, verbose=False)

if len(yolo_results) > 0 and yolo_results[0].masks is not None:
    for i, mask_data in enumerate(yolo_results[0].masks.data):
        # Convert mask to numpy array and resize to original image size
        mask = mask_data.cpu().numpy()
        mask_resized = cv2.resize(mask, (left.shape[1], left.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # FIX: Ensure mask has same dimensions as depth_map
        if mask_binary.shape != depth_map.shape:
            print(f"  Warning: Mask shape {mask_binary.shape} doesn't match depth map {depth_map.shape}")
            continue
            
        # Extract depth values for this crater - FIXED INDEXING
        mask_indices = mask_binary.astype(bool)
        crater_depths = depth_map[mask_indices]
        crater_depths = crater_depths[np.isfinite(crater_depths)]
        
        if len(crater_depths) == 0:
            continue
            
        # Calculate depth statistics
        mean_depth = np.nanmean(crater_depths)
        median_depth = np.nanmedian(crater_depths)
        min_depth = np.nanmin(crater_depths)
        max_depth = np.nanmax(crater_depths)
        
        # Get bounding box
        ys, xs = np.where(mask_binary > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
            
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        
        # Store crater data
        depth_results.append({
            "object_id": f"crater_{i}",
            "class_id": 0,
            "class_name": "Crater",
            "min_depth_m": min_depth,
            "mean_depth_m": mean_depth,
            "median_depth_m": median_depth,
            "max_depth_m": max_depth,
            "bbox": (x1, y1, x2, y2),
            "mask": mask_binary,
        })
        
        print(f"  YOLO Crater {i}: min_depth={min_depth:.2f}m, points={len(crater_depths)}")

# ===========================================================
# === STEP 2: SAM ROCK DEPTH CALCULATION ===
# ===========================================================
print("Processing SAM rocks...")
mask_dir = "sam_masks/objects"
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "mask_*.png")))

if mask_paths:
    for i, mask_path in enumerate(mask_paths):
        # Only process rock masks
        if "class1" in mask_path:
            # Load and process mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            mask_binary = (mask > 128).astype(np.uint8)
            
            # FIX: Ensure mask has same dimensions as depth_map
            if mask_binary.shape != depth_map.shape:
                print(f"  Warning: SAM mask shape {mask_binary.shape} doesn't match depth map {depth_map.shape}")
                # Resize mask to match depth_map
                mask_binary = cv2.resize(mask_binary, (depth_map.shape[1], depth_map.shape[0]))
            
            # Extract depth values for this rock - FIXED INDEXING
            mask_indices = mask_binary.astype(bool)
            rock_depths = depth_map[mask_indices]
            rock_depths = rock_depths[np.isfinite(rock_depths)]
            
            if len(rock_depths) == 0:
                continue
                
            # Calculate depth statistics
            mean_depth = np.nanmean(rock_depths)
            median_depth = np.nanmedian(rock_depths)
            min_depth = np.nanmin(rock_depths)
            max_depth = np.nanmax(rock_depths)
            
            # Get bounding box
            ys, xs = np.where(mask_binary > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
                
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            
            # Store rock data
            depth_results.append({
                "object_id": f"rock_{i}",
                "class_id": 1,
                "class_name": "Rock",
                "min_depth_m": min_depth,
                "mean_depth_m": mean_depth,
                "median_depth_m": median_depth,
                "max_depth_m": max_depth,
                "bbox": (x1, y1, x2, y2),
                "mask": mask_binary,
            })
            
            print(f"  SAM Rock {i}: min_depth={min_depth:.2f}m, points={len(rock_depths)}")

print(f"Total objects processed: {len(depth_results)}")

# ===========================================================
# === CREATE OVERLAY ON ORIGINAL IMAGE (FIXED VERSION) ===
# ===========================================================
print("\nCreating overlay on original image...")

def create_simple_overlay(original_image, depth_results):
    """
    Create an overlay showing ONLY the detected mask areas with colored outlines
    and depth labels - similar to SAM visualization
    """
    # Create a copy of the original image
    overlay = original_image.copy()
    
    # Define colors (same as point cloud coloring)
    crater_color = (0, 255, 255)  # Yellow for craters (BGR)
    rock_color = (255, 0, 0)      # Blue for rocks (BGR)
    
    # Process each object using the same masks
    for i, obj in enumerate(depth_results):
        class_name = obj['class_name']
        min_depth = obj['min_depth_m']
        mask = obj['mask']  # Same mask used for point cloud
        bbox = obj['bbox']
        
        # Choose color based on class (same as point cloud)
        color = crater_color if class_name == "Crater" else rock_color
        
        # FIX: Only draw the mask contours instead of filling the entire area
        # Find contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contour outlines (thicker for better visibility)
        cv2.drawContours(overlay, contours, -1, color, 2)
        
        # Optional: Add subtle fill with very low opacity
        colored_mask = np.zeros_like(overlay)
        for contour in contours:
            cv2.fillPoly(colored_mask, [contour], color)
        overlay = cv2.addWeighted(overlay, 0.9, colored_mask, 0.1, 0)
        
        # Add bounding box (thinner)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        
        # Add text with minimum depth
        text = f"{class_name}: {min_depth:.2f}m"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Text background (semi-transparent)
        text_bg = overlay.copy()
        cv2.rectangle(text_bg, (x1, y1 - text_size[1] - 10), 
                     (x1 + text_size[0] + 10, y1), color, -1)
        overlay = cv2.addWeighted(text_bg, 0.6, overlay, 0.4, 0)
        
        # Text
        text_y = y1 - 5
        cv2.putText(overlay, text, (x1 + 5, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    return overlay

# Alternative version: Even cleaner with just contours and labels
def create_clean_overlay(original_image, depth_results):
    """
    Clean overlay with just mask contours and depth labels - no filled areas
    """
    overlay = original_image.copy()
    
    # Define colors
    crater_color = (0, 255, 255)  # Yellow for craters (BGR)
    rock_color = (255, 0, 0)      # Blue for rocks (BGR)
    
    for i, obj in enumerate(depth_results):
        class_name = obj['class_name']
        min_depth = obj['min_depth_m']
        mask = obj['mask']
        bbox = obj['bbox']
        
        color = crater_color if class_name == "Crater" else rock_color
        
        # Draw only the contour of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 3)  # Thicker contours
        
        # Add simple text label near the object
        x1, y1, x2, y2 = bbox
        text = f"{min_depth:.2f}m"
        
        # Put text with background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(overlay, (x1, y1 - text_size[1] - 5), 
                     (x1 + text_size[0] + 5, y1), color, -1)
        cv2.putText(overlay, text, (x1 + 3, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    return overlay

# Create both versions for comparison
overlay_image_clean = create_clean_overlay(left, depth_results)
overlay_image_detailed = create_simple_overlay(left, depth_results)

# ===========================================================
# === DISPLAY THE OVERLAYS ===
# ===========================================================
plt.figure(figsize=(18, 12))

# Display original and overlays
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
plt.title('Original Image', fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(overlay_image_clean, cv2.COLOR_BGR2RGB))
plt.title('Clean Overlay: Mask Contours Only\n(Yellow=Craters, Blue=Rocks)', 
          fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(overlay_image_detailed, cv2.COLOR_BGR2RGB))
plt.title('Detailed Overlay: Contours + Labels\n(Text shows minimum depth)', 
          fontsize=14, fontweight='bold')
plt.axis('off')

# Show mask visualization
plt.subplot(2, 2, 4)
# Create a mask visualization
mask_vis = np.zeros_like(left)
for obj in depth_results:
    color = (0, 255, 255) if obj['class_name'] == "Crater" else (255, 0, 0)
    mask_vis[obj['mask'] > 0] = color
plt.imshow(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB))
plt.title('All Detected Masks\n(Yellow=Craters, Blue=Rocks)', 
          fontsize=14, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the clean overlay
cv2.imwrite("depths/object_detection_overlay_clean.png", overlay_image_clean)
cv2.imwrite("depths/object_detection_overlay_detailed.png", overlay_image_detailed)
print("✅ Overlay images saved to 'depths/' folder")

# ===========================================================
# === 3D POINT CLOUD VISUALIZATION (Open3D) ===
# ===========================================================
print("\nGenerating 3D point cloud...")

# --- Camera intrinsic parameters ---
fx = 834.06
fy = 834.06
cx = left.shape[1] / 2
cy = left.shape[0] / 2
baseline_m = 0.25

# --- Create reprojection matrix Q ---
Q = np.float32([[1, 0, 0, -cx],
                [0, -1, 0, cy],
                [0, 0, 0, -fx],
                [0, 0, 1/baseline_m, 0]])

# --- Reproject to 3D ---
points_3D = cv2.reprojectImageTo3D(disp_float, Q)
colors = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

# Create mask for valid points
depth_mask = (depth_map > 0) & (depth_map < 30.0) & np.isfinite(depth_map)
points_3D[~depth_mask] = 0.0

# --- Flatten arrays for Open3D ---
points = points_3D.reshape(-1, 3)
colors = colors.reshape(-1, 3)

# Apply mask to remove invalid points
valid_indices = depth_mask.reshape(-1)
points_valid = points[valid_indices]
colors_valid = colors[valid_indices]

print(f"Original points: {len(points)}, Valid points: {len(points_valid)}")

# --- Create Open3D point cloud ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_valid)
pcd.colors = o3d.utility.Vector3dVector(colors_valid.astype(np.float32) / 255.0)

# ===========================================================
# === HYBRID SEGMENTATION: YOLOv8 FOR CRATERS + SAM FOR ROCKS ===
# ===========================================================
print("\nApplying hybrid segmentation...")

# Create a copy of the original point cloud colors
pcd_colors = np.asarray(pcd.colors).copy()

# Define colors for different classes
class_colors = {
    0: [1.0, 1.0, 0.0],  # Yellow for Craters (YOLOv8)
    1: [0.0, 1.0, 0.0],  # Green for Rocks (SAM)
    -1: [1.0, 0.0, 1.0]  # Magenta for unknown
}

# ===========================================================
# === STEP 1: YOLOv8 CRATER SEGMENTATION ===
# ===========================================================
print("Running YOLOv8 crater segmentation...")
yolo_results = crater_model.predict(left, conf=0.25, imgsz=640, verbose=False)

crater_points_count = 0
if len(yolo_results) > 0 and yolo_results[0].masks is not None:
    # Get all crater masks from YOLO
    for i, mask_data in enumerate(yolo_results[0].masks.data):
        # Convert mask to numpy array and resize to original image size
        mask = mask_data.cpu().numpy()
        mask_resized = cv2.resize(mask, (left.shape[1], left.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(bool)
        
        # Flatten mask and apply valid indices filter
        mask_flat = mask_binary.reshape(-1)
        crater_mask_flat = mask_flat[valid_indices]
        
        # Color crater points yellow
        pcd_colors[crater_mask_flat] = class_colors[0]
        crater_points_count += np.sum(crater_mask_flat)
        
        print(f"  YOLO Crater {i+1}: colored {np.sum(crater_mask_flat)} points")

print(f"Total crater points colored: {crater_points_count}")

# ===========================================================
# === STEP 2: SAM ROCK SEGMENTATION ===
# ===========================================================
print("Loading SAM rock masks...")
mask_dir = "sam_masks/objects"
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "mask_*.png")))

rock_points_count = 0
if mask_paths:
    print(f"Found {len(mask_paths)} SAM masks, processing rocks...")
    
    # Process each SAM mask - only for rocks (class1)
    for mask_path in mask_paths:
        # Only process rock masks
        if "class1" in mask_path:
            # Load and process mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            mask_binary = (mask > 128).astype(bool)
            
            # Flatten mask and apply valid indices filter
            mask_flat = mask_binary.reshape(-1)
            rock_mask_flat = mask_flat[valid_indices]
            
            # Color rock points green (only if not already colored as crater)
            rock_points = rock_mask_flat & (np.all(pcd_colors != class_colors[0], axis=1))
            pcd_colors[rock_points] = class_colors[1]
            rock_points_count += np.sum(rock_points)
            
            print(f"  SAM Rock: colored {np.sum(rock_points)} points")
else:
    print("No SAM masks found for rocks.")

print(f"Total rock points colored: {rock_points_count}")

# ===========================================================
# === UPDATE POINT CLOUD AND VISUALIZE ===
# ===========================================================
print("\nUpdating point cloud with hybrid segmentation...")
pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

# Print segmentation summary
total_points = len(pcd_colors)
crater_percentage = (crater_points_count / total_points) * 100
rock_percentage = (rock_points_count / total_points) * 100

print(f"\n📊 SEGMENTATION SUMMARY:")
print(f"   Total 3D points: {total_points}")
print(f"   Crater points: {crater_points_count} ({crater_percentage:.1f}%)")
print(f"   Rock points: {rock_points_count} ({rock_percentage:.1f}%)")
print(f"   Background points: {total_points - crater_points_count - rock_points_count}")

# Visualize the hybrid segmented point cloud
print("\nDisplaying 3D point cloud with hybrid segmentation...")
o3d.visualization.draw_geometries([pcd], 
                             window_name="3D Point Cloud - YOLOv8 Craters + SAM Rocks", 
                             width=960, 
                             height=720,
                             point_show_normal=False)

# Optional: Save the hybrid segmented point cloud
output_pcd_path = "depths/point_cloud_hybrid_segmentation.ply"
o3d.io.write_point_cloud(output_pcd_path, pcd)
print(f"Saved hybrid segmented point cloud to: {output_pcd_path}")


# ===========================================================
# === FIXED: CALCULATE DEPTHS FOR ALL OBJECTS ===
# ===========================================================
print("\nCalculating depths for all objects...")

# Store depth results for all objects
depth_results = []

# ===========================================================
# === STEP 1: YOLOv8 CRATER DEPTH CALCULATION (FIXED) ===
# ===========================================================
print("Processing YOLOv8 craters...")
yolo_results = crater_model.predict(left, conf=0.25, imgsz=640, verbose=False)

if len(yolo_results) > 0 and yolo_results[0].masks is not None:
    for i, mask_data in enumerate(yolo_results[0].masks.data):
        # Convert mask to numpy array and resize to original image size
        mask = mask_data.cpu().numpy()
        mask_resized = cv2.resize(mask, (left.shape[1], left.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # FIX: Ensure mask has same dimensions as depth_map
        if mask_binary.shape != depth_map.shape:
            print(f"  Warning: Mask shape {mask_binary.shape} doesn't match depth map {depth_map.shape}")
            continue
            
        # Extract depth values for this crater - FIXED INDEXING
        mask_indices = mask_binary.astype(bool)
        crater_depths = depth_map[mask_indices]
        crater_depths = crater_depths[np.isfinite(crater_depths)]
        
        if len(crater_depths) == 0:
            continue
            
        # Calculate depth statistics
        mean_depth = np.nanmean(crater_depths)
        median_depth = np.nanmedian(crater_depths)
        min_depth = np.nanmin(crater_depths)
        max_depth = np.nanmax(crater_depths)
        
        # Get bounding box
        ys, xs = np.where(mask_binary > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
            
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        
        # Store crater data
        depth_results.append({
            "object_id": f"crater_{i}",
            "class_id": 0,
            "class_name": "Crater",
            "min_depth_m": min_depth,
            "mean_depth_m": mean_depth,
            "median_depth_m": median_depth,
            "max_depth_m": max_depth,
            "bbox": (x1, y1, x2, y2),
            "mask": mask_binary,
        })
        
        print(f"  YOLO Crater {i}: min_depth={min_depth:.2f}m, points={len(crater_depths)}")

# ===========================================================
# === STEP 2: SAM ROCK DEPTH CALCULATION (FIXED) ===
# ===========================================================
print("Processing SAM rocks...")
mask_dir = "sam_masks/objects"
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "mask_*.png")))

if mask_paths:
    for i, mask_path in enumerate(mask_paths):
        # Only process rock masks
        if "class1" in mask_path:
            # Load and process mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            mask_binary = (mask > 128).astype(np.uint8)
            
            # FIX: Ensure mask has same dimensions as depth_map
            if mask_binary.shape != depth_map.shape:
                print(f"  Warning: SAM mask shape {mask_binary.shape} doesn't match depth map {depth_map.shape}")
                # Resize mask to match depth_map
                mask_binary = cv2.resize(mask_binary, (depth_map.shape[1], depth_map.shape[0]))
            
            # Extract depth values for this rock - FIXED INDEXING
            mask_indices = mask_binary.astype(bool)
            rock_depths = depth_map[mask_indices]
            rock_depths = rock_depths[np.isfinite(rock_depths)]
            
            if len(rock_depths) == 0:
                continue
                
            # Calculate depth statistics
            mean_depth = np.nanmean(rock_depths)
            median_depth = np.nanmedian(rock_depths)
            min_depth = np.nanmin(rock_depths)
            max_depth = np.nanmax(rock_depths)
            
            # Get bounding box
            ys, xs = np.where(mask_binary > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
                
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            
            # Store rock data
            depth_results.append({
                "object_id": f"rock_{i}",
                "class_id": 1,
                "class_name": "Rock",
                "min_depth_m": min_depth,
                "mean_depth_m": mean_depth,
                "median_depth_m": median_depth,
                "max_depth_m": max_depth,
                "bbox": (x1, y1, x2, y2),
                "mask": mask_binary,
            })
            
            print(f"  SAM Rock {i}: min_depth={min_depth:.2f}m, points={len(rock_depths)}")

print(f"Total objects processed: {len(depth_results)}")

# ===========================================================
# === FIXED: CREATE OCCUPANCY GRID WITH EXACT MASK SHAPES ===
# ===========================================================
print("\nCreating occupancy grid with exact mask shapes...")

def create_shape_based_occupancy_grid(depth_results, depth_map, grid_resolution=0.05, max_depth=5.0, lateral_range=4.0):
    """
    Create occupancy grid that preserves exact object shapes from masks
    """
    fx = 834.06
    cx = left.shape[1] / 2
    
    depth_cells = int(max_depth / grid_resolution)
    lateral_cells = int(lateral_range / grid_resolution)
    
    # Create empty occupancy grid
    occupancy_grid = np.zeros((depth_cells, lateral_cells))
    shape_grid = np.zeros((depth_cells, lateral_cells, 3), dtype=np.uint8)
    
    print(f"Grid size: {depth_cells}x{lateral_cells} cells")
    print(f"Range: {lateral_range:.1f}m wide x {max_depth:.1f}m deep")
    print(f"Grid resolution: {grid_resolution:.2f}m per cell")
    
    # Process each object
    for obj_idx, obj in enumerate(depth_results):
        class_name = obj['class_name']
        mask = obj['mask']
        
        # Get object color
        if class_name == "Rock":
            color = np.array([255, 0, 0])  # Red for rocks
            obstacle_value = 1.0
        else:  # Crater
            color = np.array([200, 100, 0])  # Orange for craters
            obstacle_value = 0.8
        
        # Process each pixel in the mask
        ys, xs = np.where(mask > 0)
        
        print(f"  Processing {class_name} {obj_idx}: {len(xs)} pixels")
        
        for y, x in zip(ys, xs):
            # Get depth at this pixel
            depth_val = depth_map[y, x]
            if not np.isfinite(depth_val) or depth_val <= 0 or depth_val > max_depth:
                continue
            
            # Convert pixel to world coordinates
            lateral_pos = (x - cx) * depth_val / fx
            
            # Convert to grid coordinates
            depth_cell = int(depth_val / grid_resolution)
            lateral_cell = int((lateral_pos + lateral_range / 2) / grid_resolution)
            
            # Mark as occupied in grid
            if 0 <= depth_cell < depth_cells and 0 <= lateral_cell < lateral_cells:
                occupancy_grid[depth_cell, lateral_cell] = obstacle_value
                shape_grid[depth_cell, lateral_cell] = color
    
    return occupancy_grid, shape_grid

# Create the shape-based occupancy grid with 5m depth and smaller grid
try:
    occupancy_grid, shape_grid = create_shape_based_occupancy_grid(
        depth_results,
        depth_map,
        grid_resolution=0.05,  # Smaller grid cells (5cm)
        max_depth=5.0,         # Limit to 5m depth
        lateral_range=4.0      # Slightly narrower lateral range
    )
    
    # Create binary grid for navigation
    binary_grid = (occupancy_grid > 0).astype(np.uint8)
    
    print("✅ Occupancy grid created successfully!")
    
except Exception as e:
    print(f"❌ Error creating occupancy grid: {e}")
    # Create empty grids as fallback
    occupancy_grid = np.zeros((100, 80))
    shape_grid = np.zeros((100, 80, 3), dtype=np.uint8)
    binary_grid = np.zeros((100, 80), dtype=np.uint8)

# ===========================================================
# === ENHANCED VISUALIZATION WITH EXACT SHAPES (FLIPPED) ===
# ===========================================================
plt.figure(figsize=(20, 12))

# Grid coordinates (flipped for robot view from bottom)
grid_resolution = 0.05
max_depth = 5.0
lateral_range = 4.0
depth_coords = np.linspace(0, max_depth, binary_grid.shape[0])
lateral_coords = np.linspace(-lateral_range/2, lateral_range/2, binary_grid.shape[1])

# Calculate FOV
fov_angle = np.radians(75)
fov_x = np.tan(fov_angle/2) * depth_coords

# 1️⃣ Shape-Based Occupancy Grid (FLIPPED)
plt.subplot(2, 3, 1)
# Flip the grid vertically using extent and origin
plt.imshow(shape_grid, extent=[-lateral_range/2, lateral_range/2, 0, max_depth], 
           aspect='auto', origin='lower')
plt.fill_betweenx(depth_coords, -fov_x, fov_x, color='yellow', alpha=0.2, label='Robot FOV')
plt.plot(0, 0, 'bo', markersize=12, markeredgecolor='white', label='Robot')
plt.xlabel('Lateral Distance (m)')
plt.ylabel('Depth Distance (m)\nForward →')
plt.title(f'Exact Object Shapes on Occupancy Grid\n(Red=Rocks, Orange=Craters)\n5m Range, {grid_resolution*100:.0f}cm Resolution')
plt.legend()
plt.grid(True, alpha=0.3)

# 2️⃣ Binary Occupancy Grid (FLIPPED)
plt.subplot(2, 3, 2)
cmap = plt.cm.colors.ListedColormap(['green', 'red'])
plt.imshow(binary_grid, extent=[-lateral_range/2, lateral_range/2, 0, max_depth],
           cmap=cmap, aspect='auto', alpha=0.8, origin='lower')
plt.fill_betweenx(depth_coords, -fov_x, fov_x, color='yellow', alpha=0.2, label='Robot FOV')
plt.plot(0, 0, 'bo', markersize=12, markeredgecolor='white', label='Robot')

# Add object centroids
fx = 834.06
cx = left.shape[1] / 2
for obj in depth_results:
    min_depth = obj['min_depth_m']
    if min_depth > max_depth:  # Skip objects beyond 5m
        continue
    bbox = obj['bbox']
    center_x = (bbox[0] + bbox[2]) / 2
    lateral_pos = (center_x - cx) * min_depth / fx
    color = 'darkred' if obj['class_name'] == "Rock" else 'darkorange'
    marker = 's' if obj['class_name'] == "Rock" else 'o'
    plt.plot(lateral_pos, min_depth, marker=marker, color=color, markersize=8,
             markeredgecolor='white')

plt.xlabel('Lateral Distance (m)')
plt.ylabel('Depth Distance (m)\nForward →')
plt.title(f'Binary Occupancy Grid\n(Green=Free, Red=Occupied)\n5m Range, {grid_resolution*100:.0f}cm Resolution')
plt.legend()

# 3️⃣ Object Distribution (FLIPPED)
plt.subplot(2, 3, 3)
for obj in depth_results:
    class_name = obj['class_name']
    min_depth = obj['min_depth_m']
    if min_depth > max_depth:  # Skip objects beyond 5m
        continue
    bbox = obj['bbox']
    center_x = (bbox[0] + bbox[2]) / 2
    lateral_pos = (center_x - cx) * min_depth / fx
    color = 'red' if class_name == "Rock" else 'orange'
    marker = 's' if class_name == "Rock" else 'o'
    plt.scatter(lateral_pos, min_depth, c=color, marker=marker, s=70,
                label=f'{class_name}', edgecolors='black', alpha=0.7)

plt.fill_betweenx(depth_coords, -fov_x, fov_x, alpha=0.2, color='yellow', label='Robot FOV')
plt.plot(0, 0, 'bo', markersize=12, markeredgecolor='white', label='Robot')
plt.xlabel('Lateral Distance (m)')
plt.ylabel('Depth Distance (m)\nForward →')
plt.title(f'Object Distribution Map\n5m Range')
plt.legend()
plt.grid(True, alpha=0.3)

# 4️⃣ Depth Statistics (filtered for 5m range)
plt.subplot(2, 3, 4)
# Create depth histogram for all objects within 5m
depths_all = [obj['min_depth_m'] for obj in depth_results ]
classes_all = [obj['class_name'] for obj in depth_results ]

rock_depths = [d for d, c in zip(depths_all, classes_all) if c == "Rock"]
crater_depths = [d for d, c in zip(depths_all, classes_all) if c == "Crater"]

plt.hist(rock_depths, bins=10, alpha=0.7, color='red', label='Rocks', edgecolor='black')
plt.hist(crater_depths, bins=10, alpha=0.7, color='orange', label='Craters', edgecolor='black')
plt.xlabel('Minimum Depth (m)')
plt.ylabel('Number of Objects')
plt.title(f'Object Depth Distribution')
plt.axvline(x=max_depth, color='gray', linestyle='--', alpha=0.7, label='5m Limit')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===========================================================
# === SAVE RESULTS (FLIPPED VERSION) ===
# ===========================================================
print("\nSaving results...")

# Save depth statistics
df = pd.DataFrame(depth_results)
# Remove mask column for CSV (it's too large)
df_save = df.drop('mask', axis=1, errors='ignore')
df_save.to_csv("depths/object_depths_detailed.csv", index=False)
print("✅ Saved detailed object depth data")

# Save final occupancy grid (FLIPPED)
plt.figure(figsize=(12, 8))
plt.imshow(shape_grid, extent=[-lateral_range/2, lateral_range/2, 0, max_depth], 
           aspect='auto', origin='lower')
plt.fill_betweenx(depth_coords, -fov_x, fov_x, color='yellow', alpha=0.3, label='Robot FOV')
plt.plot(0, 0, 'bo', markersize=15, markeredgecolor='white', linewidth=2, label='ROBOT')

plt.xlabel('Lateral Distance (m)\n← Left | Right →', fontsize=12)
plt.ylabel('Depth Distance (m)\nForward →', fontsize=12)
plt.title(f'2D OCCUPANCY GRID - EXACT OBJECT SHAPES\n(Red=Rocks, Orange=Craters)\n5m Range, {grid_resolution*100:.0f}cm Resolution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Add distance markers (flipped)
for depth in [1, 2, 3, 4, 5]:
    plt.axhline(y=depth, color='blue', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.text(lateral_range/2 - 0.2, depth, f'{depth}m', 
             ha='right', va='bottom', color='blue', alpha=0.7)

plt.tight_layout()
plt.savefig("depths/occupancy_grid_exact_shapes.png", dpi=300, bbox_inches='tight')
plt.show()

# ===========================================================
# === NAVIGATION ANALYSIS (UPDATED FOR 5M RANGE) ===
# ===========================================================
print("\n=== NAVIGATION ANALYSIS (5m Range) ===")

total_cells = binary_grid.size
occupied_cells = np.sum(binary_grid == 1)
free_cells = np.sum(binary_grid == 0)
occupancy_ratio = occupied_cells / total_cells if total_cells > 0 else 0

print(f"Grid Analysis:")
print(f"  Total cells: {total_cells}")
print(f"  Free cells: {free_cells} ({free_cells/total_cells:.1%})")
print(f"  Occupied cells: {occupied_cells} ({occupancy_ratio:.1%})")
print(f"  Grid resolution: {grid_resolution*100:.0f}cm per cell")
print(f"  Range: {lateral_range:.1f}m wide x {max_depth:.1f}m deep")

# Count objects within 5m range
objects_within_5m = [obj for obj in depth_results if obj['min_depth_m'] <= max_depth]
print(f"\nObject Summary (within 5m):")
print(f"  Total objects: {len(objects_within_5m)}")
print(f"  Craters: {len([obj for obj in objects_within_5m if obj['class_name'] == 'Crater'])}")
print(f"  Rocks: {len([obj for obj in objects_within_5m if obj['class_name'] == 'Rock'])}")

if objects_within_5m:
    min_obj_depth = min([obj['min_depth_m'] for obj in objects_within_5m])
    max_obj_depth = max([obj['min_depth_m'] for obj in objects_within_5m])
    print(f"\nDepth Range (within 5m):")
    print(f"  Nearest object: {min_obj_depth:.2f}m")
    print(f"  Farthest object: {max_obj_depth:.2f}m")
else:
    print("\nNo objects detected within 5m range")

print(f"\n🎯 EXACT SHAPE OCCUPANCY GRID SAVED: 'depths/occupancy_grid_exact_shapes.png'")

# ===========================================================
# === SIMPLE WAYPOINT PLANNING (75° FOV CONSTRAINED) ===
# ===========================================================
print("\n=== GENERATING NAVIGATION WAYPOINTS (75° FOV) ===")

# Camera parameters
CAMERA_HFOV_DEG = 75  # Your actual camera FOV

def calculate_fov_boundaries(depth, fov_angle=np.radians(CAMERA_HFOV_DEG)):
    """
    Calculate lateral boundaries of 75° FOV at a given depth
    """
    fov_half_width = np.tan(fov_angle/2) * depth
    return -fov_half_width, fov_half_width

def find_safe_waypoints_75fov(binary_grid, grid_resolution=0.05, max_depth=5.0, lateral_range=4.0):
    """
    Simple waypoint planning that finds the safest path within 75° FOV
    """
    depth_cells, lateral_cells = binary_grid.shape
    waypoints = []
    
    print(f"Finding safest path within {CAMERA_HFOV_DEG}° camera FOV...")
    
    # Start at robot position (center)
    waypoints.append({'x': 0.0, 'y': 0.0, 'safe': True, 'safety_score': 100.0, 'within_fov': True})
    
    # Find waypoints at regular intervals
    for depth in np.arange(1.0, max_depth, 1.0):  # Every 1 meter
        depth_cell = int(depth / grid_resolution)
        
        if depth_cell >= depth_cells:
            break
        
        # Calculate 75° FOV boundaries at this depth
        fov_left, fov_right = calculate_fov_boundaries(depth)
            
        best_lateral = 0.0
        best_safety = -1
        
        # Test lateral positions ONLY within 75° FOV
        lateral_positions = np.linspace(
            max(-lateral_range/2 + 0.5, fov_left + 0.1),
            min(lateral_range/2 - 0.5, fov_right - 0.1),
            20
        )
        
        if len(lateral_positions) == 0:
            print(f"  ⚠️  No valid lateral positions within {CAMERA_HFOV_DEG}° FOV at {depth:.1f}m")
            continue
        
        for lateral in lateral_positions:
            lateral_cell = int((lateral + lateral_range/2) / grid_resolution)
            
            if not (0 <= lateral_cell < lateral_cells):
                continue
            
            # Calculate safety score
            safety_score = 0
            search_radius = int(0.5 / grid_resolution)
            
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    check_depth = depth_cell + dy
                    check_lateral = lateral_cell + dx
                    
                    if (0 <= check_depth < depth_cells and 
                        0 <= check_lateral < lateral_cells):
                        if binary_grid[check_depth, check_lateral] == 0:
                            distance = np.sqrt((dx*grid_resolution)**2 + (dy*grid_resolution)**2)
                            safety_score += max(0, 0.5 - distance)
            
            if safety_score > best_safety:
                best_safety = safety_score
                best_lateral = lateral
        
        # Add waypoint if safe
        if best_safety > 0:
            waypoints.append({
                'x': best_lateral, 
                'y': depth, 
                'safety_score': best_safety,
                'safe': True,
                'within_fov': True
            })
            print(f"  ✅ Waypoint at {depth:.1f}m: lateral={best_lateral:.2f}m, safety={best_safety:.1f}")
        else:
            print(f"  ⚠️  No safe waypoint found at {depth:.1f}m within {CAMERA_HFOV_DEG}° FOV")
    
    return waypoints

def smooth_waypoints_75fov(waypoints):
    """
    Smooth the path while keeping waypoints within 75° FOV
    """
    if len(waypoints) < 3:
        return waypoints
    
    smoothed = [waypoints[0]]
    
    for i in range(1, len(waypoints)-1):
        prev = waypoints[i-1]
        curr = waypoints[i]
        next_wp = waypoints[i+1]
        
        # Simple averaging with FOV constraint
        smoothed_x = (prev['x'] + curr['x'] + next_wp['x']) / 3
        
        # Keep within 75° FOV
        fov_left, fov_right = calculate_fov_boundaries(curr['y'])
        smoothed_x = np.clip(smoothed_x, fov_left + 0.1, fov_right - 0.1)
        
        smoothed.append({
            'x': smoothed_x,
            'y': curr['y'],
            'safety_score': curr['safety_score'],
            'safe': curr['safe'],
            'within_fov': True
        })
    
    smoothed.append(waypoints[-1])
    return smoothed

# Generate waypoints with 75° FOV constraints
waypoints = find_safe_waypoints_75fov(
    binary_grid,
    grid_resolution=0.05,
    max_depth=5.0,
    lateral_range=4.0
)

# Smooth the path
smoothed_waypoints = smooth_waypoints_75fov(waypoints)

print(f"\n✅ Waypoint Planning Complete:")
print(f"   Generated {len(smoothed_waypoints)} waypoints")
print(f"   Path length: {smoothed_waypoints[-1]['y']:.1f}m")
print(f"   Camera FOV: {CAMERA_HFOV_DEG}°")

# Display waypoint list
print(f"\n📋 WAYPOINT COORDINATES (x=lateral, y=depth):")
for i, wp in enumerate(smoothed_waypoints):
    safety = wp.get('safety_score', 100.0)
    print(f"   WP{i}: ({wp['x']:6.2f}m, {wp['y']:4.1f}m) - Safety: {safety:.1f}")

# ===========================================================
# === VISUALIZE WAYPOINTS WITH 75° FOV ===
# ===========================================================
print("\nVisualizing waypoints...")

plt.figure(figsize=(15, 10))

# 1️⃣ Waypoints on Occupancy Grid with 75° FOV
plt.subplot(2, 2, 1)
plt.imshow(shape_grid, extent=[-lateral_range/2, lateral_range/2, 0, max_depth], 
           aspect='auto', origin='lower', alpha=0.7)

# Plot 75° FOV boundaries
depth_coords = np.linspace(0, max_depth, 100)
fov_x_75 = np.tan(np.radians(CAMERA_HFOV_DEG)/2) * depth_coords
plt.fill_betweenx(depth_coords, -fov_x_75, fov_x_75, color='yellow', alpha=0.3, 
                 label=f'{CAMERA_HFOV_DEG}° Camera FOV')

# Plot waypoints and path
if smoothed_waypoints:
    waypoints_x = [wp['x'] for wp in smoothed_waypoints]
    waypoints_y = [wp['y'] for wp in smoothed_waypoints]
    
    # Plot path line
    plt.plot(waypoints_x, waypoints_y, 'g-', linewidth=4, alpha=0.8, label='Navigation Path')
    
    # Plot waypoints
    plt.scatter(waypoints_x, waypoints_y, c='blue', s=150, marker='o', 
                edgecolors='white', linewidth=3, label='Waypoints')
    
    # Plot robot start position
    plt.plot(0, 0, 'ro', markersize=20, markeredgecolor='white', linewidth=3, label='Robot Start')
    
    # Add waypoint labels
    for i, wp in enumerate(smoothed_waypoints):
        plt.annotate(f'WP{i}', (wp['x'], wp['y']), 
                    xytext=(8, 8), textcoords='offset points', 
                    fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Add safety circle
        safety_circle = plt.Circle((wp['x'], wp['y']), 0.3, color='green', alpha=0.2)
        plt.gca().add_patch(safety_circle)

plt.xlabel('Lateral Distance (m)')
plt.ylabel('Depth Distance (m)\nForward →')
plt.title(f'Navigation Waypoints with {CAMERA_HFOV_DEG}° FOV')
plt.legend()
plt.grid(True, alpha=0.3)

# 2️⃣ Binary Grid with Path
plt.subplot(2, 2, 2)
cmap = plt.cm.colors.ListedColormap(['green', 'red'])
plt.imshow(binary_grid, extent=[-lateral_range/2, lateral_range/2, 0, max_depth],
           cmap=cmap, aspect='auto', alpha=0.8, origin='lower')
plt.fill_betweenx(depth_coords, -fov_x_75, fov_x_75, color='yellow', alpha=0.2, label=f'{CAMERA_HFOV_DEG}° FOV')

if smoothed_waypoints:
    plt.plot(waypoints_x, waypoints_y, 'white', linewidth=3, alpha=0.9, label='Path')
    plt.scatter(waypoints_x, waypoints_y, c='blue', s=100, marker='s', 
                edgecolors='white', linewidth=2, label='Waypoints')

plt.xlabel('Lateral Distance (m)')
plt.ylabel('Depth Distance (m)\nForward →')
plt.title('Path on Occupancy Grid')
plt.legend()
plt.grid(True, alpha=0.3)

# 3️⃣ Safety Analysis
plt.subplot(2, 2, 3)
if smoothed_waypoints:
    depths = [wp['y'] for wp in smoothed_waypoints if wp['y'] > 0]
    safety_scores = [wp.get('safety_score', 100.0) for wp in smoothed_waypoints if wp['y'] > 0]
    
    plt.plot(depths, safety_scores, 'o-', linewidth=2, markersize=8, 
             color='purple', markerfacecolor='white', markeredgecolor='purple')
    
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Safety Threshold')
    
    plt.xlabel('Depth Distance (m)')
    plt.ylabel('Safety Score')
    plt.title('Waypoint Safety Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 4️⃣ FOV Width
plt.subplot(2, 2, 4)
depths_fov = np.linspace(0.5, max_depth, 50)
fov_widths = 2 * np.tan(np.radians(CAMERA_HFOV_DEG)/2) * depths_fov

plt.plot(depths_fov, fov_widths, 'orange', linewidth=3)
plt.xlabel('Depth Distance (m)')
plt.ylabel('FOV Width (m)')
plt.title(f'{CAMERA_HFOV_DEG}° FOV Width vs Depth')
plt.grid(True, alpha=0.3)

# Mark waypoint depths
if smoothed_waypoints:
    wp_depths = [wp['y'] for wp in smoothed_waypoints if wp['y'] > 0]
    wp_fov_widths = [2 * np.tan(np.radians(CAMERA_HFOV_DEG)/2) * d for d in wp_depths]
    plt.scatter(wp_depths, wp_fov_widths, c='red', s=50, marker='o', label='Waypoints')

plt.legend()

plt.tight_layout()
plt.show()

# ===========================================================
# === SAVE WAYPOINTS ===
# ===========================================================
print("\nSaving waypoints to file...")

# Save waypoints to CSV
waypoints_df = pd.DataFrame(smoothed_waypoints)
waypoints_df['waypoint_id'] = range(len(smoothed_waypoints))
waypoints_df.to_csv("depths/navigation_waypoints.csv", index=False)
print("✅ Saved waypoints to 'depths/navigation_waypoints.csv'")

# Save robot-readable waypoints
with open("depths/waypoints_robot.txt", "w") as f:
    f.write(f"# Robot Navigation Waypoints ({CAMERA_HFOV_DEG}° FOV)\n")
    f.write("# Format: WAYPOINT_ID, LATERAL_POS(m), DEPTH_POS(m), SAFETY_SCORE\n")
    for i, wp in enumerate(smoothed_waypoints):
        safety = wp.get('safety_score', 100.0)
        f.write(f"{i}, {wp['x']:.3f}, {wp['y']:.3f}, {safety:.1f}\n")

print("✅ Saved robot waypoints to 'depths/waypoints_robot.txt'")

# ===========================================================
# === FINAL NAVIGATION SUMMARY ===
# ===========================================================
print(f"\n🎯 NAVIGATION READY!")
print(f"   Total Waypoints: {len(smoothed_waypoints)}")
print(f"   Final Position: ({smoothed_waypoints[-1]['x']:.2f}m, {smoothed_waypoints[-1]['y']:.1f}m)")
print(f"   Camera FOV: {CAMERA_HFOV_DEG}°")

# Calculate safety statistics
safety_scores = [wp.get('safety_score', 100.0) for wp in smoothed_waypoints if wp['y'] > 0]
if safety_scores:
    avg_safety = np.mean(safety_scores)
    min_safety = min(safety_scores)
    print(f"   Average Safety: {avg_safety:.1f}")
    
    if min_safety > 10:
        print(f"   ✅ All waypoints are SAFE (min safety: {min_safety:.1f})")
    else:
        print(f"   ⚠️  Caution: Low safety waypoints (min: {min_safety:.1f})")

print(f"\n📁 Output Files:")
print(f"   - depths/navigation_waypoints.csv")
print(f"   - depths/waypoints_robot.txt")