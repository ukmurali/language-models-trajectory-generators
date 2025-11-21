# api.py
import sys
import time
import numpy as np
from PIL import Image
import config
import models
import utils
from config import OK, PROGRESS, ENDC
from config import (
    CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY,
    OPEN_GRIPPER, CLOSE_GRIPPER
)
import math
from scipy.spatial.transform import Rotation as R
import torch

from prompts.success_detection_prompt import SUCCESS_DETECTION_PROMPT


class API:
    # --- Paste inside class API (replace the old implementations) ---

    def __init__(self, args, main_connection, logger, client, langsam_model, xmem_model, device):
        # existing init code...
        self.args = args
        self.main_connection = main_connection
        self.logger = logger
        self.client = client
        self.langsam_model = langsam_model
        self.xmem_model = xmem_model
        self.device = device

        self.segmentation_texts = []
        self.segmentation_count = 0
        self.trajectory_length = 0
        self.attempted_task = False
        self.completed_task = False
        self.failed_task = False
        self.head_camera_position = None
        self.head_camera_orientation_q = None
        self.wrist_camera_position = None
        self.wrist_camera_orientation_q = None
        self.command = None

        # NEW: store the last full set of detected cubes and masks for multi-object ops
        self.last_cubes = None  # list/ndarray of shape (N, 10, 3) or None
        self.last_orientations = None  # list/ndarray of shape (N, 2) or None
        self.last_masks = None  # masks tensor/list shape (N, H, W) or None

    # --------------------------
    # Helper: choose best object
    # --------------------------
    def choose_best_object(self, cubes):
        """
        Choose the most suitable cube to grasp.
        Strategy: prefer cubes closest to the current EE XY, fallback to first.
        cubes: iterable/ndarray of shape (N,10,3)
        Returns index of chosen cube (int) and cube itself.
        """
        try:
            if cubes is None:
                return None, None
            # If single cube as ndarray (10,3)
            cubes_arr = np.asarray(cubes)
            if cubes_arr.ndim == 2 and cubes_arr.shape == (10, 3):
                return 0, cubes_arr

            # compute top-centers and distances to current EE XY
            ee = np.array(self.ee_current_position if hasattr(self, "ee_current_position") else config.ee_start_position)
            distances = []
            for i, c in enumerate(cubes_arr):
                top_center = np.mean(c[:4], axis=0)
                d = np.linalg.norm(top_center[:2] - np.array(ee[:2]))
                distances.append((d, i))
            distances.sort()
            best_idx = int(distances[0][1])
            return best_idx, cubes_arr[best_idx]
        except Exception:
            # fallback - return first
            try:
                return 0, np.asarray(cubes)[0]
            except Exception:
                return None, None

    # --------------------------
    # LLM-BASED OBJECT IDENTIFIER
    # --------------------------
    def choose_object_with_llm(self, scene_description, user_goal):
        """
        Use a language model (ChatGPT / GPT-4o) to decide which object
        the robot should segment.

        scene_description: text listing visible objects (generated manually or via vision caption)
        user_goal: natural language instruction ("pick the cup", "grab the green item")

        Returns a segmentation prompt text to pass into detect_object().
        """
        try:
            prompt = f"""
            You are controlling a robot. Based on the objects in the scene and the user's goal,
            choose EXACTLY ONE object name that should be segmented.

            Scene objects: {scene_description}
            User goal: {user_goal}

            Return ONLY the object name (example: "red cup").
            """

            llm_output = models.get_chatgpt_output(
                self.client,
                self.args.language_model,
                prompt,
                messages=[],
                role="user"
            )

            # Parse last message
            object_name = llm_output[-1]["content"].strip().lower()
            return object_name

        except Exception as e:
            print("[LLM ERROR] Falling back to user-provided segmentation text:", e)
            return None

    # --------------------------
    # Helper: classify grasp strategy
    # --------------------------
    def classify_grasp_strategy(self, cube):
        """
        Simple heuristic to decide top vs side grasp.
        cube: (10,3) array: top4, centroid, bottom4, bottom_centroid
        returns: "top" or "side"
        """
        try:
            cube = np.asarray(cube)
            # estimate width/length/height from points
            top_pts = cube[:4]
            bottom_pt = cube[5] if cube.shape[0] > 5 else cube[-1]
            width = float(np.linalg.norm(top_pts[1] - top_pts[0]))
            length = float(np.linalg.norm(top_pts[2] - top_pts[1]))
            height = float(abs(np.mean(top_pts[:, 2]) - bottom_pt[2]))
            # If height is small relative to width/length, side grasp may work.
            # But prefer top grasp for small objects (cylinders).
            if height > max(width, length) * 0.7:
                return "top"
            # If one horizontal dimension is very small (<0.08) suggest side (finger pinch)
            if min(width, length) < 0.08:
                return "side"
            return "top"
        except Exception:
            return "top"

    # --------------------------
    # Helper: compute grasp poses
    # --------------------------
    def compute_grasp_poses(self, cube, strategy="top"):
        """
        Compute hover (safe) and touch (grasp) positions and a default quaternion.
        Applies TCP offset so gripper fingers align correctly with the grasp target.
        Returns: (grasp_hover, grasp_touch, quat)
        """
        cube = np.asarray(cube)
        top_center = np.mean(cube[:4], axis=0)
        x, y, z = float(top_center[0]), float(top_center[1]), float(top_center[2])

        # TCP offset in meters (negative value means TCP is above fingers)
        tcp_z_offset = -0.12  # Adjust this based on your robot's gripper TCP

        if strategy == "side":
            # side grasp uses a slightly higher hover and more horizontal approach
            hover_z = z + 0.18
            touch_z = max(z + 0.10, 0.05)
        else:
            # top grasp conservative offsets
            hover_z = z + 0.12
            touch_z = max(z + 0.02, 0.05)

        # Apply TCP offset to align gripper fingers with the object surface
        grasp_hover = [x, y, hover_z + tcp_z_offset]
        grasp_touch = [x, y, touch_z + tcp_z_offset]

        # default: gripper pointing down (x,y,z,w)
        quat = R.from_euler("xyz", [math.pi, 0, 0]).as_quat().tolist()
        return grasp_hover, grasp_touch, quat

    # --------------------------
    # Helper: stable grasp orientation from mask (fallback)
    # --------------------------
    def compute_stable_grasp_orientation(self, mask):
        """
        Given a single mask (H,W) tensor or numpy array, compute a simple orientation.
        For now, return gripper down quaternion. Future: use PCA on mask to compute yaw.
        """
        try:
            # If mask is a torch Tensor, convert to numpy
            if hasattr(mask, "cpu"):
                try:
                    import torch
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = np.array(mask)
                except Exception:
                    mask_np = np.array(mask)
            else:
                mask_np = np.array(mask)

            # PCA on mask to find major axis (2D) -> compute yaw
            coords = np.argwhere(mask_np)
            if coords.shape[0] >= 10:
                # y,x ordering from argwhere -> convert to image coords (c,r)
                mean = coords.mean(axis=0)
                cov = np.cov(coords.T)
                eigvals, eigvecs = np.linalg.eig(cov)
                principal = eigvecs[:, np.argmax(eigvals)]
                # principal is in image coords (row, col) -> convert to world yaw sign
                yaw = math.atan2(principal[0], principal[1])
                # build quaternion as downward with yaw about z
                quat = R.from_euler("zyx", [yaw, 0, math.pi]).as_quat().tolist()
                return quat
        except Exception:
            pass
        # fallback: gripper pointing down (x,y,z,w)
        return R.from_euler("xyz", [math.pi, 0, 0]).as_quat().tolist()

    # --------------------------
    # Main: detect single object (refactored, robust)
    # --------------------------
    def detect_object(self, segmentation_text):
        """
        Detects an object and returns the grasp info dict.
        """
        print('segmentation_text', segmentation_text)
        # Capture images from env
        self.logger.info(PROGRESS + "Capturing head and wrist camera images..." + ENDC)
        self.main_connection.send([CAPTURE_IMAGES])
        (
            head_camera_position,
            head_camera_orientation_q,
            wrist_camera_position,
            wrist_camera_orientation_q,
            env_msg
        ) = self.main_connection.recv()
        self.logger.info(env_msg)

        self.head_camera_position = head_camera_position
        self.head_camera_orientation_q = head_camera_orientation_q
        self.wrist_camera_position = wrist_camera_position
        self.wrist_camera_orientation_q = wrist_camera_orientation_q

        # load images saved by env
        rgb = Image.open(config.rgb_image_head_path).convert("RGB")
        depth = Image.open(config.depth_image_head_path).convert("L")
        # depth_array = np.array(depth) / 255.0
        depth_array = utils.load_depth_meters(config.depth_image_head_path, depth_scale_guess=1.5)

        # first-time XMem mask
        if self.segmentation_count == 0:
            Image.fromarray(np.zeros_like(depth_array)).convert("L").save(config.xmem_input_path)

        # segmentation
        prompts = [segmentation_text]
        self.logger.info(PROGRESS + "Segmenting head camera image..." + ENDC)
        model_preds, boxes, seg_texts = models.get_langsam_output(
            rgb, self.langsam_model, prompts, self.segmentation_count
        )
        self.logger.info(OK + "Finished segmenting head camera image!" + ENDC)

        masks = utils.get_segmentation_mask(model_preds, config.segmentation_threshold)
        # masks expected shape: list or tensor (N,H,W)

        # bounding cubes (world coords)
        cubes, orientations = utils.get_bounding_cube_from_point_cloud(
            rgb, masks, depth_array,
            self.head_camera_position, self.head_camera_orientation_q,
            self.segmentation_count
        )

        # save masks for XMem tracking (non-fatal)
        try:
            utils.save_xmem_image(masks)
        except Exception:
            self.logger.debug("Warning: save_xmem_image failed")

        # visualize cubes (non-fatal)
        try:
            if cubes is not None and len(cubes) > 0:
                self.main_connection.send([ADD_BOUNDING_CUBES, cubes])
                [env_msg] = self.main_connection.recv()
                self.logger.info(env_msg)
        except Exception:
            self.logger.debug("Warning: failed to send bounding cubes to env")

        # Update last detections (for detect_all_objects)
        self.last_cubes = np.asarray(cubes) if cubes is not None else None
        self.last_orientations = np.asarray(orientations) if orientations is not None else None
        self.last_masks = masks

        self.segmentation_count += 1

        # --------------------------
        # No objects found
        # --------------------------
        if self.last_cubes is None or len(self.last_cubes) == 0:
            print("[WARNING] No objects detected.")
            return None

        # Choose best object index
        best_idx, cube = self.choose_best_object(self.last_cubes)
        if cube is None:
            print("[WARNING] No valid cube chosen.")
            return None

        # Determine strategy and poses
        strategy = self.classify_grasp_strategy(cube)
        hover, touch, quat_from_cube = self.compute_grasp_poses(cube, strategy)

        # compute orientation (use mask PCA if available)
        mask_for_obj = None
        try:
            if self.last_masks is not None:
                # last_masks may be list-like; prefer same index
                mask_for_obj = self.last_masks[best_idx] if hasattr(self.last_masks, "__len__") and len(
                    self.last_masks) > best_idx else None
        except Exception:
            mask_for_obj = None

        orientation = self.compute_stable_grasp_orientation(
            mask_for_obj) if mask_for_obj is not None else quat_from_cube

        # compute top_center for position field
        top_center = np.mean(cube[:4], axis=0).tolist()

        # return {
        #     "label": segmentation_text,
        #     "position": top_center,
        #     "grasp_hover": [float(hover[0]), float(hover[1]), float(hover[2])],
        #     "grasp_touch": [float(touch[0]), float(touch[1]), float(touch[2])],
        #     "orientation": orientation,
        #     "masks": self.last_masks
        # }

        touch = [touch[0], touch[1], touch[2]]
        hover = [hover[0], hover[1], hover[2]]

        # debug prints
        print("Segmentation:", segmentation_text)
        print("position:", top_center)
        print("strategy:", strategy)
        print("grasp_hover:", hover)
        print("grasp_touch:", touch)
        print("orientation:", orientation)

        return {
            "label": segmentation_text,
            "position": top_center,
            "grasp_hover": [hover[0], hover[1], hover[2]],
            "grasp_touch": [touch[0], touch[1], touch[2]],
            "orientation": orientation,
            "masks": self.last_masks
        }

    # --------------------------
    # Multi-object detection helper
    # --------------------------
    def detect_all_objects(self, name):
        """
        Returns grasp info for all detected cubes (list).
        """
        # Run single detection to populate last_* caches
        info = self.detect_object(name)
        if info is None or self.last_cubes is None:
            return []

        results = []
        cubes = np.asarray(self.last_cubes)
        masks = self.last_masks

        for idx, cube in enumerate(cubes):
            try:
                top_pts = cube[:4]
                top_center = np.mean(top_pts, axis=0)
                x, y, z = float(top_center[0]), float(top_center[1]), float(top_center[2])

                hover = [x, y, z + 0.12]
                touch = [x, y, max(z + 0.02, 0.05)]
                quat = R.from_euler("xyz", [math.pi, 0, 0]).as_quat().tolist()

                mask_for_obj = None
                if masks is not None and hasattr(masks, "__len__") and len(masks) > idx:
                    mask_for_obj = masks[idx]

                orientation = self.compute_stable_grasp_orientation(mask_for_obj) if mask_for_obj is not None else quat

                results.append({
                    "label": name,
                    "position": top_center.tolist(),
                    "grasp_hover": hover,
                    "grasp_touch": touch,
                    "orientation": orientation,
                    "masks": mask_for_obj
                })
            except Exception:
                continue

        return results

    # ---------------------------
    # Utilities
    # ---------------------------
    def clamp_workspace(self, pos):
        """Clamp into reasonable Sawyer workspace (meters)."""
        x = float(np.clip(pos[0], -0.6, 0.6))
        y = float(np.clip(pos[1], -0.6, 0.8))
        z = float(np.clip(pos[2], 0.05, 1.5))
        return [x, y, z]

    def safe_point(self, pos, dz_hover=0.10, clamp=True):
        """Return safe approach point above pos (pos is [x,y,z] or dict{'pos':...})."""
        if isinstance(pos, dict) and "pos" in pos:
            pos = pos["pos"]
        x, y, z = pos
        if clamp:
            x, y, z = self.clamp_workspace([x, y, z])
        return [float(x), float(y), float(z + dz_hover)]

    def normalize_orient(self, orient):
        """Ensure quaternion is list of 4 floats [x,y,z,w]."""
        if orient is None:
            return None
        return [float(v) for v in orient]

    # ---------------------------
    # TRAJECTORY + EXECUTION
    # ---------------------------
    def execute_trajectory(self, trajectory):
        """
        Normalize and send a trajectory to the environment.
        Accepts:
            - list of dicts {"pos":.., "orient":..}
            - list of tuples (pos, orient)
            - list of [x,y,z] (orient None)
        Returns True on successful send, False on error.
        """
        # simple validation
        if not isinstance(trajectory, (list, tuple)) or len(trajectory) == 0:
            self.logger.error("[ERROR] Invalid or empty trajectory received.")
            return False

        canonical = []

        for wp in trajectory:
            # ——————————————————————————
            # Extract or normalize waypoint
            # ——————————————————————————
            if isinstance(wp, dict):
                raw_pos = wp.get("pos") or wp.get("position") or wp.get("hover") or wp.get("touch")
                raw_orient = wp.get("orient") or wp.get("orientation")
            elif isinstance(wp, (list, tuple)) and len(wp) == 2:
                raw_pos, raw_orient = wp
            else:
                self.logger.warning(f"[WARN] Unsupported waypoint format: {wp}")
                continue

            if raw_pos is None or len(raw_pos) != 3:
                self.logger.warning(f"[WARN] Invalid waypoint pos: {raw_pos}")
                continue

            # Keep VISUALIZATION coordinate BEFORE modifications
            vis_pos = [float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2])]

            # ——————————————————————————
            # APPLY workspace clamp, TCP offset, etc.
            # (This determines actual robot movement)
            # ——————————————————————————
            control_pos = self.clamp_workspace(raw_pos)

            if raw_orient is not None:
                control_orient = self.normalize_orient(raw_orient)
            else:
                control_orient = None

            # ——————————————————————————
            # Append both robot control + visualization positions
            # ——————————————————————————
            canonical.append({
                "pos": control_pos,  # used for robot motion
                "orient": control_orient,
                "vis_pos": vis_pos  # used ONLY for drawing green trajectory
            })

        if len(canonical) == 0:
            self.logger.error("[ERROR] No valid waypoints after normalization.")
            return False

        try:
            self.logger.info(PROGRESS + f"Adding {len(canonical)} trajectory points..." + ENDC)
            self.main_connection.send([ADD_TRAJECTORY_POINTS, canonical])

            self.logger.info(PROGRESS + "Executing generated trajectory..." + ENDC)
            self.main_connection.send([EXECUTE_TRAJECTORY, canonical])

            self.trajectory_length += len(canonical)
            self.logger.info(PROGRESS + "Trajectory execution command sent successfully." + ENDC)
            return True
        except Exception as e:
            self.logger.exception(f"[ERROR] Unexpected error in execute_trajectory: {e}")
            return False

    # ---------------------------
    # GRIPPER
    # ---------------------------
    def open_gripper(self):
        self.logger.info(PROGRESS + "Opening gripper..." + ENDC)
        try:
            self.main_connection.send([OPEN_GRIPPER])
        except Exception as e:
            self.logger.warning(f"[WARN] open_gripper send failed: {e}")

    def close_gripper(self):
        self.logger.info(PROGRESS + "Closing gripper..." + ENDC)
        try:
            self.main_connection.send([CLOSE_GRIPPER])
        except Exception as e:
            self.logger.warning(f"[WARN] close_gripper send failed: {e}")

    # ---------------------------
    # SUCCESS CHECK (MASK IOU)
    # ---------------------------
    def mask_iou(self, maskA, maskB):
        """Compute IoU between two boolean masks (numpy)."""
        A = np.asarray(maskA, dtype=bool)
        B = np.asarray(maskB, dtype=bool)
        inter = np.logical_and(A, B).sum()
        union = np.logical_or(A, B).sum()
        if union == 0:
            return 0.0
        return float(inter) / float(union)

    def segmentation_for_label(self, label):
        """
        Run a fresh segmentation for `label` on the current camera image.
        Returns masks (torch bool tensor) or None.
        """
        # capture fresh images
        self.main_connection.send([CAPTURE_IMAGES])
        (
            head_camera_position,
            head_camera_orientation_q,
            wrist_camera_position,
            wrist_camera_orientation_q,
            env_msg
        ) = self.main_connection.recv()
        self.logger.info(env_msg)

        rgb = Image.open(config.rgb_image_head_path).convert("RGB")
        self.logger.info(PROGRESS + f"Re-segmenting for '{label}'..." + ENDC)
        model_preds, boxes, _ = models.get_langsam_output(rgb, self.langsam_model, [label], self.segmentation_count)
        masks = utils.get_segmentation_mask(model_preds, config.segmentation_threshold)
        return masks

    def success_check_after_grasp(self, original_mask, label):
        """
        Returns True if object is successfully removed from table (i.e., IoU small).
        original_mask: torch bool [H,W] or numpy
        """
        try:
            new_masks = self.segmentation_for_label(label)
            if new_masks is None or len(new_masks) == 0:
                # if segmentation yields none, likely moved -> success
                return True

            # pick first mask
            new_mask = new_masks[0]
            # convert to numpy bool
            orig = original_mask.detach().cpu().numpy() if isinstance(original_mask, torch.Tensor) else np.asarray(
                original_mask)
            if orig.ndim == 3:
                orig = orig[0]
            new = new_mask.detach().cpu().numpy() if isinstance(new_mask, torch.Tensor) else np.asarray(new_mask)
            if new.ndim == 3:
                new = new[0]

            iou = self.mask_iou(orig, new)
            self.logger.info(PROGRESS + f"Mask IoU after grasp for '{label}': {iou:.3f}" + ENDC)
            return iou <= self.success_iou_threshold
        except Exception as e:
            self.logger.warning(f"[WARN] success_check failed: {e}")
            # conservative: assume failure so regrasp logic can run
            return False

    def task_completed(self):

        if self.attempted_task:

            self.completed_task = True

        else:

            self.logger.info(PROGRESS + "Waiting to execute all generated trajectories..." + ENDC)
            self.main_connection.send([config.TASK_COMPLETED])
            (env_connection_message) = self.main_connection.recv()
            self.logger.info(env_connection_message)

            self.logger.info(PROGRESS + "Generating XMem output..." + ENDC)
            masks = models.get_xmem_output(self.xmem_model, self.device, self.trajectory_length)
            self.logger.info(OK + "Finished generating XMem output!" + ENDC)

            num_objects = len(np.unique(masks[0])) - 1

            new_prompt = SUCCESS_DETECTION_PROMPT.replace("[INSERT TASK]", self.command)
            new_prompt += "\n"

            self.logger.info(PROGRESS + "Calculating object bounding cubes..." + ENDC)

            for object in range(1, num_objects + 1):

                object_positions = []
                object_orientations = []

                idx_offset = 0

                for i, mask in enumerate(masks):

                    rgb_image = Image.open(
                        config.rgb_image_trajectory_path.format(step=i * config.xmem_output_every)).convert("RGB")
                    depth_image = Image.open(
                        config.depth_image_trajectory_path.format(step=i * config.xmem_output_every)).convert("L")
                    depth_array = np.array(depth_image) / 255.

                    object_mask = mask.copy()
                    object_mask[object_mask != object] = False
                    object_mask[object_mask == object] = True
                    object_mask = torch.Tensor(object_mask)

                    bounding_cubes, orientations = utils.get_bounding_cube_from_point_cloud(rgb_image, [object_mask],
                                                                                            depth_array,
                                                                                            self.head_camera_position,
                                                                                            self.head_camera_orientation_q,
                                                                                            object - 1)

                    if len(bounding_cubes) == 0:

                        self.logger.info("No bounding cube found: removed.")
                        idx_offset += 1

                    else:

                        [bounding_cube] = bounding_cubes
                        [orientation] = orientations
                        position = bounding_cube[4]
                        orientation = orientation[0]
                        orientation = np.mod(orientation + math.pi, 2 * math.pi) - math.pi

                        object_positions.append(position)

                        if i == 0:

                            object_orientations.append(orientation)

                        else:

                            previous_orientation = object_orientations[i - 1 - idx_offset]
                            possible_orientations = np.array(
                                [np.mod(orientation + i * math.pi / 2 + math.pi, 2 * math.pi) - math.pi for i in
                                 range(4)])
                            circular_difference = np.minimum(np.abs(possible_orientations - previous_orientation),
                                                             2 * math.pi - np.abs(
                                                                 possible_orientations - previous_orientation))
                            min_index = np.argmin(circular_difference)
                            orientation = possible_orientations[min_index]
                            object_orientations.append(orientation)

                new_prompt += self.segmentation_texts[object - 1] + " trajectory positions and orientations:\n"
                new_prompt += "Positions:\n"
                new_prompt += str(np.around(
                    [position for p, position in enumerate(object_positions) if p % config.xmem_lm_input_every == 0],
                    3)) + "\n"
                new_prompt += "Orientations:\n"
                new_prompt += str(np.around([orientation for o, orientation in enumerate(object_orientations) if
                                             o % config.xmem_lm_input_every == 0], 3)) + "\n"
                new_prompt += "\n"

            self.logger.info(OK + "Finished calculating object bounding cubes!" + ENDC)

            self.attempted_task = True

            messages = []

            self.logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
            messages = models.get_chatgpt_output(self.client, self.args.language_model, new_prompt, messages, "system",
                                                 file=sys.stderr)
            self.logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

            code_block = messages[-1]["content"].split("```python")

            task_completed = self.task_completed
            task_failed = self.task_failed

            for block in code_block:
                if len(block.split("```")) > 1:
                    code = block.split("```")[0]
                    exec(code)

    def task_failed(self):

        self.failed_task = True

        self.logger.info(PROGRESS + "Resetting environment..." + ENDC)
        self.main_connection.send([config.RESET_ENVIRONMENT])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self.segmentation_count = 0
        self.trajectory_length = 0
        self.segmentation_texts = []
        self.attempted_task = False
