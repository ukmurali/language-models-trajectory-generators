import pybullet as p
import numpy as np
import pybullet_data
import time
import config
from robot import Robot
from config import OK, PROGRESS, FAIL, ENDC
from config import CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY, OPEN_GRIPPER, \
    CLOSE_GRIPPER, TASK_COMPLETED, RESET_ENVIRONMENT


class Environment:

    def __init__(self, args):
        self.mode = args.mode
        self.can_id = None
        self.grasp_constraint_id = None

    def load(self):
        p.resetDebugVisualizerCamera(config.camera_distance, config.camera_yaw, config.camera_pitch,
                                     config.camera_target_position)

        object_start_position = config.object_start_position
        object_start_orientation_q = p.getQuaternionFromEuler(config.object_start_orientation_e)
        self.can_id = p.loadURDF("ycb_assets/002_master_chef_can.urdf", object_start_position,
                                 object_start_orientation_q, useFixedBase=False, globalScaling=config.global_scaling)

        if self.mode == "default":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    def attach_can_to_gripper(self, robot):
        # Check whether end-effector contacts the can
        contacts = p.getContactPoints(bodyA=robot.id, bodyB=self.can_id, linkIndexA=robot.ee_index)
        if contacts:
            contact = contacts[0]
            if self.grasp_constraint_id is None:
                self.grasp_constraint_id = p.createConstraint(
                    robot.id, robot.ee_index, self.can_id, -1,
                    p.JOINT_FIXED,
                    [0, 0, 0], contact[6], contact[7])
            return True
        return False

    def release_can_from_gripper(self):
        if self.grasp_constraint_id is not None:
            p.removeConstraint(self.grasp_constraint_id)
            self.grasp_constraint_id = None

    def update(self):
        p.stepSimulation()
        time.sleep(config.control_dt)


def run_simulation_environment(args, env_connection, logger):
    # Environment set-up
    logger.info(PROGRESS + "Setting up environment..." + ENDC)

    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf", physicsClientId=physics_client)

    env = Environment(args)
    env.load()

    robot = Robot(args)
    robot.move(env, robot.ee_start_position, robot.ee_start_orientation_e, gripper_open=True, is_trajectory=False)

    env_connection_message = OK + "Finished setting up environment!" + ENDC
    env_connection.send([env_connection_message])

    while True:

        if env_connection.poll():

            env_connection_received = env_connection.recv()
            if env_connection_received[0] == CAPTURE_IMAGES:

                _, _ = robot.get_camera_image("head", env, save_camera_image=True,
                                              rgb_image_path=config.rgb_image_trajectory_path.format(step=0),
                                              depth_image_path=config.depth_image_trajectory_path.format(step=0))
                head_camera_position, head_camera_orientation_q = robot.get_camera_image("head", env,
                                                                                         save_camera_image=True,
                                                                                         rgb_image_path=config.rgb_image_head_path,
                                                                                         depth_image_path=config.depth_image_head_path)
                wrist_camera_position, wrist_camera_orientation_q = robot.get_camera_image("wrist", env,
                                                                                           save_camera_image=True,
                                                                                           rgb_image_path=config.rgb_image_wrist_path,
                                                                                           depth_image_path=config.depth_image_wrist_path)

                env_connection_message = OK + "Finished capturing head camera image!" + ENDC
                env_connection.send(
                    [head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q,
                     env_connection_message])

            elif env_connection_received[0] == ADD_BOUNDING_CUBES:

                bounding_cubes_world_coordinates = env_connection_received[1]

                for bounding_cube_world_coordinates in bounding_cubes_world_coordinates:
                    p.addUserDebugLine(bounding_cube_world_coordinates[0], bounding_cube_world_coordinates[1],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[1], bounding_cube_world_coordinates[2],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[2], bounding_cube_world_coordinates[3],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[3], bounding_cube_world_coordinates[0],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[5], bounding_cube_world_coordinates[6],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[6], bounding_cube_world_coordinates[7],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[7], bounding_cube_world_coordinates[8],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[8], bounding_cube_world_coordinates[5],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[0], bounding_cube_world_coordinates[5],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[1], bounding_cube_world_coordinates[6],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[2], bounding_cube_world_coordinates[7],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugLine(bounding_cube_world_coordinates[3], bounding_cube_world_coordinates[8],
                                       [0, 1, 0], lifeTime=0)

                    p.addUserDebugPoints(bounding_cube_world_coordinates,
                                         [[0, 1, 0]] * len(bounding_cube_world_coordinates), pointSize=5, lifeTime=0)

                env_connection_message = OK + "Finished adding bounding cubes to the environment!" + ENDC

                env_connection.send([env_connection_message])

            elif env_connection_received[0] == ADD_TRAJECTORY_POINTS:

                trajectory = env_connection_received[1]

                for wp in trajectory:

                    # World position used for visualization
                    vis = wp.get("vis_pos") or wp.get("pos")

                    # Robot control position
                    control_pos = wp.get("pos")

                    orient = wp.get("orient") or wp.get("orientation")
                    if orient and len(orient) == 4:
                        euler = p.getEulerFromQuaternion(orient)
                    else:
                        euler = robot.ee_start_orientation_e

                    # ---- DRAW VISUALIZATION TRAJECTORY ----
                    p.addUserDebugLine(
                        vis,
                        [vis[0], vis[1], vis[2] + 0.02],
                        [0, 1, 0],
                        2
                    )

                    # ---- MOVE ROBOT ----
                    robot.move(env, control_pos, euler, gripper_open=robot.gripper_open, is_trajectory=True)

                # Step simulation
                for _ in range(100):
                    env.update()

                env_connection.send([OK + "Finished executing generated trajectory!" + ENDC])

            elif env_connection_received[0] == EXECUTE_TRAJECTORY:
                trajectory = env_connection_received[1]

                def to_pos_and_euler(wp):
                    pos, orient = None, None
                    if isinstance(wp, dict):
                        pos = wp.get("pos") or wp.get("position")
                        orient = wp.get("orient") or wp.get("orientation")
                    elif isinstance(wp, (list, tuple)):
                        if len(wp) == 2:
                            pos, orient = wp
                        elif len(wp) >= 3:
                            pos = list(wp[:3])
                    if orient and len(orient) == 4:
                        euler = p.getEulerFromQuaternion(orient)
                    elif orient is not None:
                        euler = (0, 0, float(orient))  # assume yaw scalar
                    else:
                        euler = robot.ee_start_orientation_e
                    return list(map(float, pos)), list(map(float, euler))

                for wp in trajectory:
                    pos, euler = to_pos_and_euler(wp)
                    try:
                        robot.move(env, pos, euler, gripper_open=robot.gripper_open, is_trajectory=True)
                    except Exception as e:
                        print(f"[Env] Warning: robot.move failed for waypoint: {pos}, {euler}, error: {e}")

                for _ in range(100):  # simulate steps during motion
                    env.update()

                env_connection.send([OK + "Finished executing generated trajectory!" + ENDC])

            elif env_connection_received[0] == OPEN_GRIPPER:

                ee_current_position = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[0]
                ee_current_orientation_q = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[1]
                ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)

                robot.move(env, ee_current_position, ee_current_orientation_e, gripper_open=True, is_trajectory=False)

                robot.gripper_open = True

                logger.info(OK + "Finished opening gripper!" + ENDC)

            elif env_connection_received[0] == CLOSE_GRIPPER:

                ee_current_position = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[0]
                ee_current_orientation_q = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[1]
                ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)

                robot.move(env, ee_current_position, ee_current_orientation_e, gripper_open=False, is_trajectory=False)

                robot.gripper_open = False

                logger.info(OK + "Finished closing gripper!" + ENDC)

            elif env_connection_received[0] == TASK_COMPLETED:

                env_connection_message = OK + "Finished executing all generated trajectories!" + ENDC
                env_connection.send([env_connection_message])

            elif env_connection_received[0] == RESET_ENVIRONMENT:

                robot.move(env, robot.ee_start_position, robot.ee_start_orientation_e, gripper_open=True,
                           is_trajectory=False)
                robot.gripper_open = True
                robot.trajectory_step = 1

                for _ in range(100):
                    env.update()

                env_connection_message = OK + "Finished resetting environment!" + ENDC
                env_connection.send([env_connection_message])

        env.update()
