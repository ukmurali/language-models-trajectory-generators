import numpy as np
import sys
import torch
import math
import config
import models
import utils
from PIL import Image
from prompts.success_detection_prompt import SUCCESS_DETECTION_PROMPT
from config import OK, PROGRESS, FAIL, ENDC
from config import (
    CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY,
    OPEN_GRIPPER, CLOSE_GRIPPER, TASK_COMPLETED, RESET_ENVIRONMENT
)


class API:
    def __init__(self, args, main_connection, logger, client, langsam_model, xmem_model, device):
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

    # ---------------------------------------------------------------
    # DETECT OBJECT
    # ---------------------------------------------------------------
    def detect_object(self, segmentation_text):
        """Capture images, segment objects, compute bounding cubes, and log details."""

        self.logger.info(PROGRESS + "Capturing head and wrist camera images..." + ENDC)
        self.main_connection.send([CAPTURE_IMAGES])
        (
            head_camera_position,
            head_camera_orientation_q,
            wrist_camera_position,
            wrist_camera_orientation_q,
            env_connection_message
        ) = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self.head_camera_position = head_camera_position
        self.head_camera_orientation_q = head_camera_orientation_q
        self.wrist_camera_position = wrist_camera_position
        self.wrist_camera_orientation_q = wrist_camera_orientation_q

        rgb_image_head = Image.open(config.rgb_image_head_path).convert("RGB")
        depth_image_head = Image.open(config.depth_image_head_path).convert("L")
        depth_array = np.array(depth_image_head) / 255.0

        if self.segmentation_count == 0:
            # Create blank initial mask for XMem
            Image.fromarray(np.zeros_like(depth_array)).convert("L").save(config.xmem_input_path)

        # Prepare segmentation prompt
        segmentation_prompts = [segmentation_text]

        self.logger.info(PROGRESS + "Segmenting head camera image..." + ENDC)
        model_predictions, boxes, predicted_texts = models.get_langsam_output(
            rgb_image_head, self.langsam_model, segmentation_prompts, self.segmentation_count
        )
        self.logger.info(OK + "Finished segmenting head camera image!" + ENDC)

        # Use returned phrases if available
        segmentation_texts = predicted_texts if predicted_texts else segmentation_prompts

        # Convert to binary masks
        masks = utils.get_segmentation_mask(model_predictions, config.segmentation_threshold)

        # Compute bounding cubes
        bounding_cubes_world_coordinates, bounding_cubes_orientations = utils.get_bounding_cube_from_point_cloud(
            rgb_image_head,
            masks,
            depth_array,
            self.head_camera_position,
            self.head_camera_orientation_q,
            self.segmentation_count
        )

        # Save masks for XMem tracking
        utils.save_xmem_image(masks)

        # Extend stored segmentation names
        self.segmentation_texts.extend(segmentation_texts)

        # Handle no cubes found
        if (
            not isinstance(bounding_cubes_world_coordinates, np.ndarray)
            or bounding_cubes_world_coordinates.size == 0
        ):
            self.logger.info("⚠️ No bounding cubes detected.")
            self.segmentation_count += 1
            return

        # Add cubes to simulation environment
        self.logger.info(PROGRESS + "Adding bounding cubes to the environment..." + ENDC)
        self.main_connection.send([ADD_BOUNDING_CUBES, bounding_cubes_world_coordinates])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        # Normalize cubes to iterable
        cubes = bounding_cubes_world_coordinates
        if cubes.ndim == 2 and cubes.shape == (10, 3):
            cubes = cubes[np.newaxis, ...]  # single cube

        # Loop through each detected cube
        for i, cube in enumerate(cubes):
            if cube.ndim != 2 or cube.shape[1] != 3 or cube.shape[0] < 6:
                self.logger.warning(f"⚠️ Skipping malformed cube {i} with shape {cube.shape}")
                continue

            # Pick a safe label
            label = (
                segmentation_texts[i]
                if i < len(segmentation_texts)
                else f"object_{i + 1}"
            )

            # Adjust cube depth
            center_idx = 4 if cube.shape[0] > 4 else cube.shape[0] - 1
            cube[center_idx, 2] -= config.bounding_cube_depth_offset

            # Compute dimensions safely
            try:
                width = np.linalg.norm(cube[1] - cube[0])
                length = np.linalg.norm(cube[2] - cube[1])
                height = np.linalg.norm(cube[5] - cube[0])
            except Exception:
                width = length = height = float("nan")

            print(f"📦 Position of {label}:", np.round(cube[center_idx], 3).tolist())
            print("Dimensions:")
            print("Width:", np.round(width, 3))
            print("Length:", np.round(length, 3))
            print("Height:", np.round(height, 3))

            # Orientation if available
            if i < len(bounding_cubes_orientations):
                width_theta, length_theta = bounding_cubes_orientations[i]
                width_theta = np.round(width_theta, 3)
                length_theta = np.round(length_theta, 3)

                if np.isnan(width) or np.isnan(length):
                    print("Orientation (width):", width_theta)
                    print("Orientation (length):", length_theta, "\n")
                elif width < length:
                    print("Orientation along shorter side (width):", width_theta)
                    print("Orientation along longer side (length):", length_theta, "\n")
                else:
                    print("Orientation along shorter side (length):", length_theta)
                    print("Orientation along longer side (width):", width_theta, "\n")
            else:
                print("Orientation data unavailable for this cube.\n")

        self.segmentation_count += 1

    # ---------------------------------------------------------------
    # TRAJECTORY EXECUTION
    # ---------------------------------------------------------------
    def execute_trajectory(self, trajectory):
        self.logger.info(PROGRESS + "Adding trajectory points to the environment..." + ENDC)
        self.main_connection.send([ADD_TRAJECTORY_POINTS, trajectory])

        self.logger.info(PROGRESS + "Executing generated trajectory..." + ENDC)
        self.main_connection.send([EXECUTE_TRAJECTORY, trajectory])

        self.trajectory_length += len(trajectory)

    # ---------------------------------------------------------------
    # GRIPPER CONTROL
    # ---------------------------------------------------------------
    def open_gripper(self):
        self.logger.info(PROGRESS + "Opening gripper..." + ENDC)
        self.main_connection.send([OPEN_GRIPPER])

    def close_gripper(self):
        self.logger.info(PROGRESS + "Closing gripper..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])

    # ---------------------------------------------------------------
    # TASK COMPLETION
    # ---------------------------------------------------------------
    def task_completed(self):
        if self.attempted_task:
            self.completed_task = True
            return

        self.logger.info(PROGRESS + "Waiting to execute all generated trajectories..." + ENDC)
        self.main_connection.send([TASK_COMPLETED])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self.logger.info(PROGRESS + "Generating XMem output..." + ENDC)
        masks = models.get_xmem_output(self.xmem_model, self.device, self.trajectory_length)
        self.logger.info(OK + "Finished generating XMem output!" + ENDC)

        num_objects = len(np.unique(masks[0])) - 1

        new_prompt = SUCCESS_DETECTION_PROMPT.replace("[INSERT TASK]", self.command) + "\n"
        self.logger.info(PROGRESS + "Calculating object bounding cubes..." + ENDC)

        for obj_idx in range(1, num_objects + 1):
            positions, orientations = [], []
            idx_offset = 0

            for i, mask in enumerate(masks):
                rgb_image = Image.open(config.rgb_image_trajectory_path.format(step=i * config.xmem_output_every)).convert("RGB")
                depth_image = Image.open(config.depth_image_trajectory_path.format(step=i * config.xmem_output_every)).convert("L")
                depth_array = np.array(depth_image) / 255.0

                object_mask = torch.tensor(mask == obj_idx, dtype=torch.bool)

                cubes, cube_orients = utils.get_bounding_cube_from_point_cloud(
                    rgb_image, [object_mask], depth_array,
                    self.head_camera_position, self.head_camera_orientation_q, obj_idx - 1
                )

                if len(cubes) == 0:
                    self.logger.info(f"No bounding cube found for object {obj_idx} at step {i}")
                    idx_offset += 1
                    continue

                cube = cubes[0]
                orient = cube_orients[0][0]

                pos = cube[4]
                positions.append(pos)

                if i == 0:
                    orientations.append(orient)
                else:
                    prev = orientations[i - 1 - idx_offset]
                    options = np.array([
                        np.mod(orient + k * math.pi / 2 + math.pi, 2 * math.pi) - math.pi for k in range(4)
                    ])
                    diff = np.minimum(np.abs(options - prev), 2 * math.pi - np.abs(options - prev))
                    orientations.append(options[np.argmin(diff)])

            new_prompt += f"{self.segmentation_texts[obj_idx - 1]} trajectory positions and orientations:\n"
            new_prompt += "Positions:\n" + str(np.round(positions, 3)) + "\n"
            new_prompt += "Orientations:\n" + str(np.round(orientations, 3)) + "\n\n"

        self.logger.info(OK + "Finished calculating object bounding cubes!" + ENDC)
        self.attempted_task = True

        self.logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
        messages = models.get_chatgpt_output(
            self.client, self.args.language_model, new_prompt, [], "system", file=sys.stderr
        )
        self.logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

        code_blocks = [b.split("```")[0] for b in messages[-1]["content"].split("```python") if "```" in b]
        for code in code_blocks:
            exec(code)

    # ---------------------------------------------------------------
    # TASK FAILED
    # ---------------------------------------------------------------
    def task_failed(self):
        self.failed_task = True
        self.logger.info(PROGRESS + "Resetting environment..." + ENDC)
        self.main_connection.send([RESET_ENVIRONMENT])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        # Reset states
        self.segmentation_count = 0
        self.trajectory_length = 0
        self.segmentation_texts = []
        self.attempted_task = False
