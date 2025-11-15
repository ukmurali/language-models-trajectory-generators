import pybullet as p
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import math
import config
from PIL import Image
from torchvision.utils import save_image
from shapely.geometry import MultiPoint, Polygon, polygon


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


def save_xmem_image(masks: torch.Tensor):
    """
    Combine multiple binary masks into a single labeled mask image for XMem.
    Supports [N,H,W], [N,1,H,W], [1,N,H,W], [H,W]. Saves to config.xmem_input_path.
    """
    masks = _ensure_masks_3d_bool(masks)  # [N,H,W] bool
    if masks.numel() == 0:
        print("⚠️ No masks to save for XMem — skipping.")
        return

    N, H, W = masks.shape
    print(f"🧪 Saving {N} masks for XMem of size {H}x{W}")

    xmem_array = np.zeros((H, W), dtype=np.uint8)
    masks_np = masks.detach().cpu().numpy()  # [N,H,W], bool

    for idx in range(N):
        xmem_array[masks_np[idx]] = idx + 1  # label indices 1..N

    Image.fromarray(xmem_array, mode="L").save(config.xmem_input_path)
    print(f"✅ XMem input mask saved to: {config.xmem_input_path}")


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


def get_world_point_world_frame(camera_position, camera_orientation_q, camera, image, point):
    # point = (c, r, depth)
    image_width, image_height = image.size
    K, Rt = get_intrinsics_extrinsics(image_height, camera_position, camera_orientation_q)

    # shift pixel coords to center
    c, r, depth = point
    pixel_point = np.array([[c - (image_width / 2.0)],
                            [(image_height / 2.0) - r],
                            [1.0]], dtype=np.float64)

    # adjust for camera mount convention
    if camera == "wrist":
        pixel_point = np.array([[pixel_point[1, 0]], [pixel_point[0, 0]], [pixel_point[2, 0]]], dtype=np.float64)
    elif camera == "head":
        pixel_point = np.array([[-pixel_point[1, 0]], [-pixel_point[0, 0]], [pixel_point[2, 0]]], dtype=np.float64)

    world_point_camera_frame = (np.linalg.inv(K) @ pixel_point) * float(depth)
    world_point_world_frame = Rt @ np.vstack((world_point_camera_frame, np.array([1.0], dtype=np.float64)))
    return world_point_world_frame.squeeze()[:-1]


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

def get_bounding_cube_from_point_cloud(image, masks, depth_array, camera_position, camera_orientation_q, segmentation_count):
    """
    image: PIL.Image (for size)
    masks: torch.Tensor [N,H,W] bool OR compatible (will be normalized)
    depth_array: np.ndarray [H,W] depth in meters normalized to (0..1] or real depth (ensure same as used in world projection)
    camera_position/orientation: for projection
    """
    image_width, image_height = image.size
    masks = _ensure_masks_3d_bool(masks)  # [N,H,W] bool

    bounding_cubes = []
    bounding_cubes_orientations = []

    for i, mask in enumerate(masks):  # mask: [H,W] bool
        # Save mask preview (torchvision expects [C,H,W] float in [0,1])
        mask_to_save = mask.float().unsqueeze(0)
        save_image(mask_to_save, config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i))

        # Use mask pixels directly instead of expensive pointPolygonTest
        mask_np = mask.detach().cpu().numpy()
        ys, xs = np.where(mask_np)  # pixel coordinates inside mask

        if xs.size == 0:
            continue  # empty mask

        # Gather (c, r, depth) tuples, guard depth bounds
        depth_clip = np.clip(depth_array, 0, None)
        depths = depth_clip[ys, xs]
        # Filter out invalid depths (0 or NaN/inf)
        valid = np.isfinite(depths) & (depths > 0)
        xs_v = xs[valid]
        ys_v = ys[valid]
        depths_v = depths[valid]
        if xs_v.size == 0:
            continue

        # Project to world points
        contour_world_points = [
            get_world_point_world_frame(camera_position, camera_orientation_q, "head", image, (int(c), int(r), float(d)))
            for c, r, d in zip(xs_v, ys_v, depths_v)
        ]
        contour_world_points = np.asarray(contour_world_points, dtype=np.float64)
        if contour_world_points.shape[0] < 4:
            continue

        # Heights
        z_vals = contour_world_points[:, 2]
        max_z_coordinate = float(np.max(z_vals))
        min_z_coordinate = float(np.min(z_vals))

        # Keep top surface band
        top_band = max_z_coordinate - float(config.point_cloud_top_surface_filter)
        top_surface_world_points = contour_world_points[z_vals > top_band]
        if top_surface_world_points.shape[0] < 4:
            # fallback: use all points if too few on top
            top_surface_world_points = contour_world_points

        # Oriented minimum rectangle in XY
        rect = MultiPoint([tuple(pt[:2]) for pt in top_surface_world_points]).minimum_rotated_rectangle
        if isinstance(rect, Polygon):
            rect = polygon.orient(rect, sign=-1)
            box = np.asarray(rect.exterior.coords[:-1], dtype=np.float64)  # (4,2)

            # Order such that index 0 is min x
            box = np.roll(box, -int(np.argmin(box[:, 0])), axis=0)

            # Build cube: 4 top + center, 4 bottom + center  -> each is [x,y,z]
            box_top = [list(pt) + [max_z_coordinate] for pt in box]
            box_btm = [list(pt) + [min_z_coordinate] for pt in box]
            box_top.append(list(np.mean(box_top, axis=0)))
            box_btm.append(list(np.mean(box_btm, axis=0)))
            bounding_cubes.append(box_top + box_btm)

            # Orientations along width/length (in XY)
            width_theta = math.atan2(box[1][1] - box[0][1], box[1][0] - box[0][0])
            length_theta = math.atan2(box[2][1] - box[1][1], box[2][0] - box[1][0])
            bounding_cubes_orientations.append([width_theta, length_theta])

    bounding_cubes = np.array(bounding_cubes, dtype=np.float64) if len(bounding_cubes) else np.zeros((0,))
    return bounding_cubes, bounding_cubes_orientations
