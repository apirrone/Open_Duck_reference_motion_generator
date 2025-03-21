import placo
from placo_utils.visualization import robot_frame_viz, robot_viz
import numpy as np
import time

# Loading a robot
robot = placo.RobotWrapper(
    "open_duck_reference_motion_generator/robots/open_duck_mini_v2/open_duck_mini_v2.urdf"
)
# Initializing the kinematics solver
solver = placo.KinematicsSolver(robot)

trunk_pose = np.eye(4)
trunk_pose[:3, 3] = [0, 0, 0.26]
trunk_task = solver.add_frame_task("trunk", trunk_pose)
trunk_task.configure("trunk", "soft", 1.0, 1.0)

head_pose = trunk_pose.copy()
head_pose[:3, 3] += [0, 0, 0.1]
head_task = solver.add_frame_task("head", head_pose)
head_task.configure("head", "soft", 1.0, 1.0)

left_foot_pose = np.eye(4)
left_foot_pose[:3, 3] = [0, 0.08, 0]
left_foot_task = solver.add_frame_task("left_foot", left_foot_pose)
left_foot_task.configure("left_foot", "soft", 1.0, 1.0)


right_foot_pose = np.eye(4)
right_foot_pose[:3, 3] = [0, -0.08, 0]
right_foot_task = solver.add_frame_task("right_foot", right_foot_pose)
right_foot_task.configure("right_foot", "soft", 1.0, 1.0)

viz = robot_viz(robot)

FPS = 50
episode = {
    "LoopMode": "Wrap",
    "FPS": FPS,
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Joints": [],
    "Vel_x": [],
    "Vel_y": [],
    "Yaw": [],
    "Placo": [],
    "Frame_offset": [],
    "Frame_size": [],
    "Frames": [],
    "MotionWeight": 1,
}


start_t = time.time()
i = 0
contacts = [1, 1]
while True:
    t = time.time() - start_t
    if t < 0.3:
        trunk_pose[:3, 3] += [0, 0, -0.002]
        head_pose[:3, 3] += [0, 0, -0.002]

    if t > 0.3 and t < 0.4:
        trunk_pose[:3, 3] += [0, 0, 0.007]
        head_pose[:3, 3] += [0, 0, 0.01]
        contacts = [0, 0]

    if t > 0.38 and t < 0.48:
        left_foot_pose[:3, 3] += [0, 0, 0.007]
        right_foot_pose[:3, 3] += [0, 0, 0.007]
        contacts = [0, 0]

    if t > 0.6:
        contacts = [1, 1]

    if t > 2.0:
        trunk_pose[:3, 3] = [0, 0, 0.26]
        left_foot_pose[:3, 3] = [0, 0.08, 0]
        right_foot_pose[:3, 3] = [0, -0.08, 0]
        head_pose[:3, 3] = trunk_pose[:3, 3] + [0, 0, 0.1]
        contacts = [1, 1]
    if t > 3.0:
        start_t = time.time()

    print(contacts)

    left_foot_task.T_world_frame = left_foot_pose
    right_foot_task.T_world_frame = right_foot_pose
    trunk_task.T_world_frame = trunk_pose
    head_task.T_world_frame = head_pose

    # head_pose = trunk_pose.copy()
    # head_pose[:3, 3] += [0, 0, 0.1]
    # Updating kinematics computations (frames, jacobians, etc.)
    robot.update_kinematics()
    # Solving the IK
    solver.solve(True)

    if i % 5 == 0:
        viz.display(robot.state.q)
        robot_frame_viz(robot, "trunk")
        robot_frame_viz(robot, "left_foot")
        robot_frame_viz(robot, "right_foot")
        robot_frame_viz(robot, "head")


        # episode["Frames"].append(
        #     root_position
        #     + root_orientation_quat
        #     + joints_positions
        #     + left_toe_pos
        #     + right_toe_pos
        #     + world_linear_vel
        #     + world_angular_vel
        #     + joints_vel
        #     + left_toe_vel
        #     + right_toe_vel
        #     + foot_contacts
        # )

    time.sleep(0.01)
    i += 1
