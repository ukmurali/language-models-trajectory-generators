import os

import pybullet as p
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import math
import config
from PIL import Image
from torchvision.utils import save_image
from shapely.geometry import MultiPoint, Polygon


# ---------- Mask utilities ----------

def _ensure_masks_3d_bool(masks: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
    """
    Normalize masks to shape [N, H, W] with dtype=bool.
    Accepts [H, W], [N, H, W], [1, N, H, W], [N, 1, H, W], float/bool/int.
    Optionally thresholds float masks.
    """
    if masks is None or (torch.is_tensor(masks) and masks.numel() == 0):
        return torch.zeros((0,), dtype=torch.bool)

    if not torch.is_tensor(masks):
        raise ValueError("get_segmentation_mask/save_xmem_image expect a torch.Tensor")

    # Squeeze any leading singleton dims (e.g., [1, N, H, W] -> [N, H, W])
    while masks.ndim > 3 and masks.shape[0] == 1:
        masks = masks.squeeze(0)

    # If [N,1,H,W] collapse channel dim
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]

    # If now [H, W], add N dim
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    if masks.ndim != 3:
        # As a last resort, try to squeeze all singleton dims
        masks = masks.squeeze()
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.ndim != 3:
            raise ValueError(f"Masks must be [N,H,W]; got shape {tuple(masks.shape)}")

    # Threshold if requested or if float
    if threshold is not None and masks.dtype.is_floating_point:
        masks = masks > threshold

    # If still float/int and no threshold provided, treat >0 as foreground
    if not masks.dtype == torch.bool:
        if masks.dtype.is_floating_point:
            masks = masks > 0.5
        else:
            masks = masks != 0

    return masks.bool()


def get_segmentation_mask(masks: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert predicted masks to binary [N, H, W] bool.
    """
    if masks is None or (torch.is_tensor(masks) and masks.numel() == 0):
        print("⚠️ No masks provided to get_segmentation_mask()")
        return torch.zeros((0,), dtype=torch.bool)

    bin_masks = _ensure_masks_3d_bool(masks, threshold=threshold)
    print(f"✅ Generated {bin_masks.shape[0]} binary masks with threshold {threshold}")
    return bin_masks


def safe_save_mask(mask, path):
    # mask: torch tensor [H,W] or [1,H,W] or numpy float/bool
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(mask)

    # convert bool → uint8
    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8) * 255

    # convert float → uint8
    if np.issubdtype(mask.dtype, np.floating):
        mask = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)

    Image.fromarray(mask, mode="L").save(path)


# ---------- Geometry / projection ----------

def get_intrinsics_extrinsics(image_height, camera_position, camera_orientation_q):
    # Keep user's convention: principal point handled by centered pixel coords later
    fov = (config.fov / 360.0) * 2.0 * math.pi
    f = image_height / (2.0 * math.tan(fov / 2.0))
    K = np.array([[f, 0, 0],
                  [0, f, 0],
                  [0, 0, 1]], dtype=np.float64)

    R = np.array(p.getMatrixFromQuaternion(camera_orientation_q), dtype=np.float64).reshape(3, 3)
    Rt = np.hstack((R, np.array(camera_position, dtype=np.float64).reshape(3, 1)))
    Rt = np.vstack((Rt, np.array([0, 0, 0, 1], dtype=np.float64)))
    return K, Rt


def get_world_point_world_frame(camera_position, camera_orientation_q, camera, image, point, K=None, Rt=None):
    """
    Convert pixel (u, v, depth_m) -> world frame 3D point (meters).
    Prefer using Rt (4x4) if given; otherwise compute R from camera_orientation_q (PyBullet quaternion).
    - image: PIL.Image or object with .size
    - point: (u, v, depth_m) where u is column (x pixel), v is row (y pixel)
    - K: optional camera intrinsics 3x3. If cx==0 or cy==0, this function will override with image center.
    - Rt: optional 4x4 camera-to-world transform (preferred)
    Returns: np.array([X,Y,Z]) in world frame (meters)
    """
    u, v, depth = point
    W, H = image.size
    depth = float(depth)  # in meters

    # Validate/use K
    if K is None:
        # fallback focal length estimate in pixels
        fx = fy = 221.70250337
        cx = W / 2.0
        cy = H / 2.0
    else:
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        # If principal point is zero (bad), set to image center
        if cx == 0.0 and cy == 0.0:
            cx = W / 2.0
            cy = H / 2.0

    # Camera coordinates (pinhole model)
    Zc = depth
    Xc = (u - cx) * Zc / fx
    Yc = (v - cy) * Zc / fy
    cam_point = np.array([Xc, Yc, Zc], dtype=np.float64).reshape(3, 1)

    # Use Rt if possible (camera->world)
    if Rt is not None:
        Rcw = np.array(Rt[:3, :3], dtype=np.float64)
        tcw = np.array(Rt[:3, 3], dtype=np.float64).reshape(3, 1)
        world_point = (Rcw @ cam_point) + tcw
        return world_point.flatten()

    # Otherwise use quaternion (PyBullet quaternion order assumed)
    try:
        import pybullet as p
        
        T = np.array(camera_position).reshape(3, 1)
        world_point = (Rcw @ cam_point) + T
        return world_point.flatten()

        R_wc = np.array(p.getMatrixFromQuaternion(camera_orientation_q)).reshape(3, 3)
        R_cw = R_wc.T  # inverse rotation

        T_wc = np.array(camera_position).reshape(3, 1)

        world_point = (R_cw @ cam_point) + T_wc
    except Exception:
        # last resort: treat camera position as translation only
        T = np.array(camera_position).reshape(3, 1)
        world_point = cam_point + T
        return world_point.flatten()


# ---------- Contour helper (kept but not relied upon) ----------

def get_max_contour(image_gray, image_width, image_height):
    """
    Returns the largest contour by area from a uint8 grayscale mask.
    """
    # Binary threshold
    _, thresh = cv.threshold(image_gray, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Choose contour with max area
    return max(contours, key=cv.contourArea)


# ---------- Main function: bounding cube ----------

# the patched function
def get_bounding_cube_from_point_cloud(image, masks, depth_array, camera_position, camera_orientation_q,
                                       segmentation_count):
    """
    Robust bounding-cube extraction from image mask pixels + depth.

    Returns:
      bounding_cubes: np.ndarray shape (num_cubes, 10, 3)
        order per cube: top_corners[0..3], geometric_centroid, bottom_corners[0..3], bottom_centroid
      bounding_cubes_orientations: list of [width_theta, length_theta]
    Notes:
      - depth_array can be normalized (0..1) or metric (meters). This function will
        try to detect which it is by checking max value (> 1.5 -> meters).
      - get_world_point_world_frame(...) is called per point; keep it as your project provides.
    """
    masks = _ensure_masks_3d_bool(masks)  # ensure [N,H,W] bool

    bounding_cubes = []
    bounding_cubes_orientations = []

    # Decide whether depth_array is normalized or metric
    # Heuristic: if max depth > 1.5, treat as meters; else treat as normalized 0..1

    for i, mask in enumerate(masks):  # mask: [H,W] bool
        # save preview (non-fatal)
        try:
            mask_to_save = mask.float().unsqueeze(0)
            save_image(mask_to_save, config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i))
        except Exception:
            pass

        mask_np = mask.detach().cpu().numpy()
        ys, xs = np.where(mask_np)  # r,c pixel coords

        if xs.size == 0:
            continue  # empty mask

        # sample depths at mask pixels (clip negatives)
        depths_raw = depth_array[ys, xs]
        # if normalized, treat as [0..1]; keep as-is to match project's pipeline
        depths = np.array(depths_raw, dtype=np.float64)
        # filter invalid entries (zero/NaN)
        valid = np.isfinite(depths) & (depths > 0)
        xs_v = xs[valid]
        ys_v = ys[valid]
        depths_v = depths[valid]
        if xs_v.size == 0:
            continue

        # log depth range for debugging
        try:
            d_min, d_max = float(np.min(depths_v)), float(np.max(depths_v))
            print(f"[get_bounding_cube] mask {i} depth range: min={d_min:.6f} max={d_max:.6f}")
        except Exception:
            pass

        # Project pixel -> world points using existing helper.
        # get_world_point_world_frame expects (c, r, depth) where depth is in same units your pipeline uses.
        contour_world_points = []
        for c, r, d in zip(xs_v, ys_v, depths_v):
            try:
                pt = get_world_point_world_frame(camera_position, camera_orientation_q, "head", image,
                                                 (int(c), int(r), float(d)))
                contour_world_points.append(pt)
            except Exception:
                # skip problematic pixels
                continue

        contour_world_points = np.asarray(contour_world_points, dtype=np.float64)
        if contour_world_points.shape[0] < 4:
            continue

        # Heuristic top/bottom heights
        z_vals = contour_world_points[:, 2]
        max_z_coordinate = float(np.max(z_vals))
        min_z_coordinate = float(np.min(z_vals))

        # pick top-surface points (within top filter)
        top_band = max_z_coordinate - float(config.point_cloud_top_surface_filter)
        top_surface_world_points = contour_world_points[z_vals >= top_band]
        if top_surface_world_points.shape[0] < 4:
            top_surface_world_points = contour_world_points

        # Build a rotated minimum-area rectangle in XY
        xy_pts = [tuple(pt[:2]) for pt in top_surface_world_points]
        try:
            mp = MultiPoint(xy_pts)
            rect = mp.minimum_rotated_rectangle  # shapely geometry (Polygon)
            if not isinstance(rect, Polygon):
                # fallback: bounding box
                rect = Polygon(mp.convex_hull.envelope.exterior.coords)
            box_coords = np.array(rect.exterior.coords[:-1], dtype=np.float64)  # (4,2)
        except Exception:
            # fallback: use axis-aligned bounding box of top_surface_world_points
            xs_box = top_surface_world_points[:, 0]
            ys_box = top_surface_world_points[:, 1]
            xmin, xmax = float(np.min(xs_box)), float(np.max(xs_box))
            ymin, ymax = float(np.min(ys_box)), float(np.max(ys_box))
            box_coords = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float64)

        # ensure we have 4 points
        if box_coords.shape[0] != 4:
            # attempt to compute convex hull -> sample 4 points
            hull = MultiPoint(xy_pts).convex_hull
            if isinstance(hull, Polygon):
                coords = np.array(hull.exterior.coords[:-1], dtype=np.float64)
                if coords.shape[0] >= 4:
                    # pick 4 evenly spaced hull points
                    idxs = np.round(np.linspace(0, coords.shape[0] - 1, 4)).astype(int)
                    box_coords = coords[idxs]
                else:
                    continue
            else:
                continue

        # Stable ordering: sort points by angle around centroid (CCW), then rotate so first point has min x
        centroid_xy = np.mean(box_coords, axis=0)
        angles = np.arctan2(box_coords[:, 1] - centroid_xy[1], box_coords[:, 0] - centroid_xy[0])
        order = np.argsort(angles)
        box = box_coords[order]

        # rotate so index 0 is min x (stable deterministic ordering)
        min_x_idx = int(np.argmin(box[:, 0]))
        if min_x_idx != 0:
            box = np.roll(box, -min_x_idx, axis=0)

        # Form top and bottom corners (x,y,z)
        box_top = [[float(pt[0]), float(pt[1]), max_z_coordinate] for pt in box]
        box_btm = [[float(pt[0]), float(pt[1]), min_z_coordinate] for pt in box]

        # geometric centroid is mean of the 8 corners
        all_corners = np.vstack([np.array(box_top)[:, :3], np.array(box_btm)[:, :3]])
        geometric_centroid = np.mean(all_corners, axis=0).tolist()

        top_centroid = [geometric_centroid[0], geometric_centroid[1], max_z_coordinate]
        bottom_centroid = [geometric_centroid[0], geometric_centroid[1], min_z_coordinate]

        cube_points = box_top + [geometric_centroid] + box_btm + [bottom_centroid]
        bounding_cubes.append(cube_points)

        # orientation angles
        width_theta = math.atan2(box[1][1] - box[0][1], box[1][0] - box[0][0])
        length_theta = math.atan2(box[2][1] - box[1][1], box[2][0] - box[1][0])
        bounding_cubes_orientations.append([width_theta, length_theta])

    bounding_cubes = np.array(bounding_cubes, dtype=np.float64) if len(bounding_cubes) else np.zeros((0,))
    print(f"[get_bounding_cube] returning {bounding_cubes.shape} cubes")
    return bounding_cubes, bounding_cubes_orientations


def load_depth_meters(depth_path, depth_scale_guess=1.5):
    """
    Load a depth PNG and return a float32 depth map in meters.
    Heuristics:
      - uint8: assume 0..255 maps linearly to [0, depth_scale_guess] meters
      - uint16: assume values are millimeters -> convert to meters
      - float32/64: assume values already in meters
    Returns: depth_m (H,W) float32
    """
    img = Image.open(depth_path)
    arr = np.array(img)

    if arr.dtype == np.uint8:
        depth_norm = arr.astype(np.float32) / 255.0
        depth_m = depth_norm * float(depth_scale_guess)
    elif arr.dtype == np.uint16:
        # typical simulators store depth in mm
        depth_m = arr.astype(np.float32) / 1000.0
    elif arr.dtype in (np.float32, np.float64):
        depth_m = arr.astype(np.float32)
    else:
        # fallback: normalize by max
        depth_norm = arr.astype(np.float32) / float(arr.max() if arr.max() > 0 else 1.0)
        depth_m = depth_norm * float(depth_scale_guess)

    return depth_m


def calibrate_depth_scale(depth_path, sample_pixel, known_world_z, K, Rt, image, depth_scale_grid=(0.5, 3.0, 0.01)):
    """
    Brute-force search for scale factor that makes the reprojected world Z at sample_pixel
    match known_world_z. Returns best_scale (meters).
    - sample_pixel: (u,v) pixel coordinates
    - known_world_z: scalar z in world frame (meters)
    - K, Rt: intrinsics and camera-to-world Rt (4x4)
    - image: PIL image for size
    - depth_scale_grid: (min, max, step)
    """
    from math import isfinite
    u, v = sample_pixel
    best_scale = None
    best_err = float("inf")
    min_s, max_s, step = depth_scale_grid
    s = min_s
    while s <= max_s:
        depth_map = load_depth_meters(depth_path, depth_scale_guess=s)
        depth_m = float(depth_map[int(v), int(u)])
        world_pt = get_world_point_world_frame(
            camera_position=Rt[:3, 3].tolist(),
            camera_orientation_q=None,
            camera=None,
            image=image,
            point=(u, v, depth_m),
            K=K,
            Rt=Rt
        )
        err = abs(world_pt[2] - known_world_z)
        if err < best_err:
            best_err = err
            best_scale = s
        s += step
    return best_scale, best_err


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (radians) to quaternion (x, y, z, w)
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qx, qy, qz, qw]


def save_xmem_image(masks, path="./images/xmem_input.png"):
    """
    Saves mask(s) into a single grayscale PNG for XMem.
    If multiple masks exist, merges them into one binary mask.
    """
    if masks is None:
        return False

    # Convert torch -> numpy
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()

    # masks shape can be: [N,H,W] or [H,W]
    if masks.ndim == 3:
        merged = np.any(masks, axis=0).astype(np.uint8) * 255
    elif masks.ndim == 2:
        merged = masks.astype(np.uint8) * 255
    else:
        return False

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    Image.fromarray(merged, mode="L").save(path)
    return True

