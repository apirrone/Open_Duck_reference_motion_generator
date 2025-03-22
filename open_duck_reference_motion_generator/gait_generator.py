import argparse
import json
import os
import time
import webbrowser
import threading
import numpy as np
from placo_utils.visualization import footsteps_viz, robot_frame_viz, robot_viz
from scipy.spatial.transform import Rotation as R

from placo_walk_engine import PlacoWalkEngine


def open_browser():
    try:
        webbrowser.open_new("http://127.0.0.1:7000/static/")
    except:
        print("Failed to open the default browser. Trying Google Chrome.")
        try:
            webbrowser.get("google-chrome").open_new("http://127.0.0.1:7000/static/")
        except:
            print(
                "Failed to open Google Chrome. Make sure it's installed and accessible."
            )


class RoundingFloat(float):
    __repr__ = staticmethod(lambda x: format(x, ".5f"))


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, required=True)
script_path = os.path.dirname(os.path.abspath(__file__))
parser.add_argument(
    "-o", "--output_dir", type=str, default=os.path.join(script_path, "recordings")
)
parser.add_argument("--dx", type=float, default=0)
parser.add_argument("--dy", type=float, default=0)
parser.add_argument("--dtheta", type=float, default=0)
parser.add_argument("--double_support_ratio", type=float, default=None)
parser.add_argument("--startend_double_support_ratio", type=float, default=None)
parser.add_argument("--planned_timesteps", type=float, default=None)
parser.add_argument("--replan_timesteps", type=float, default=None)
parser.add_argument("--walk_com_height", type=float, default=None)
parser.add_argument("--walk_foot_height", type=float, default=None)
parser.add_argument("--walk_trunk_pitch", type=float, default=None)
parser.add_argument("--walk_foot_rise_ratio", type=float, default=None)
parser.add_argument("--single_support_duration", type=float, default=None)
parser.add_argument("--single_support_timesteps", type=float, default=None)
parser.add_argument("--foot_length", type=float, default=None)
parser.add_argument("--feet_spacing", type=float, default=None)
parser.add_argument("--zmp_margin", type=float, default=None)
parser.add_argument("--foot_zmp_target_x", type=float, default=None)
parser.add_argument("--foot_zmp_target_y", type=float, default=None)
parser.add_argument("--walk_max_dtheta", type=float, default=None)
parser.add_argument("--walk_max_dy", type=float, default=None)
parser.add_argument("--walk_max_dx_forward", type=float, default=None)
parser.add_argument("--walk_max_dx_backward", type=float, default=None)
parser.add_argument("-l", "--length", type=int, default=10)
parser.add_argument("-m", "--meshcat_viz", action="store_true", default=False)
parser.add_argument(
    "--duck",
    choices=["go_bdx", "open_duck_mini", "open_duck_mini_v2", "sigmaban2019"],
    help="Duck type",
    required=True,
)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--preset", type=str, default="")
parser.add_argument(
    "-s",
    "--skip_warmup",
    action="store_true",
    default=False,
    help="don't record warmup motion",
)
parser.add_argument(
    "--stand",
    action="store_true",
    default=False,
    help="hack to record a standing pose",
)
args = parser.parse_args()
args.hardware = True

FPS = 50  # 50 for mujoco playground, 30 for AWD
MESHCAT_FPS = 20
DISPLAY_MESHCAT = args.meshcat_viz

# For IsaacGymEnvs (OUTDATED)
# [root position, root orientation, joint poses (e.g. rotations)]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14]

# For AWD and amp for hardware
# [root position, root orientation, joint poses (e.g. rotations), target toe positions, linear velocity, angular velocity, joint velocities, target toe velocities]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, l_toe_x, l_toe_y, l_toe_z, r_toe_x, r_toe_y, r_toe_z, lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z, j1_vel, j2_vel, j3_vel, j4_vel, j5_vel, j6_vel, j7_vel, j8_vel, j9_vel, j10_vel, j11_vel, j12_vel, j13_vel, j14_vel, l_toe_vel_x, l_toe_vel_y, l_toe_vel_z, r_toe_vel_x, r_toe_vel_y, r_toe_vel_z]

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
if args.debug:
    episode["Debug_info"] = []

script_path = os.path.dirname(os.path.abspath(__file__))
robot = args.duck
robot_urdf = f"{robot}.urdf"
asset_path = os.path.join(script_path, f"robots/{robot}")

preset_filename = args.preset
filename = os.path.join(asset_path, "placo_defaults.json")
if preset_filename:
    if os.path.exists(preset_filename):
        filename = preset_filename
    else:
        print(f"No such file: {preset_filename}")
with open(filename, "r") as f:
    gait_parameters = json.load(f)
    print(f"gait_parameters {gait_parameters}")

args.dx = gait_parameters["dx"]
args.dy = gait_parameters["dy"]
args.dtheta = gait_parameters["dtheta"]

pwe = PlacoWalkEngine(asset_path, robot_urdf, gait_parameters)

first_joints_positions = list(pwe.get_angles().values())
first_T_world_fbase = pwe.robot.get_T_world_fbase()
first_T_world_leftFoot = pwe.robot.get_T_world_left()
first_T_world_rightFoot = pwe.robot.get_T_world_right()

pwe.set_traj(args.dx, args.dy, args.dtheta + 0.00955)
if DISPLAY_MESHCAT:
    viz = robot_viz(pwe.robot)
    threading.Timer(1, open_browser).start()
DT = 0.001
start = time.time()

last_record = 0
last_meshcat_display = 0
prev_root_position = [0, 0, 0]
prev_root_orientation_quat = None
prev_root_orientation_euler = [0, 0, 0]
prev_left_toe_pos = [0, 0, 0]
prev_right_toe_pos = [0, 0, 0]
prev_joints_positions = None
i = 0
prev_initialized = False
avg_x_lin_vel = []
avg_y_lin_vel = []
avg_yaw_vel = []
added_frame_info = False
# center_y_pos = None
# center_y_pos = -(pwe.parameters.feet_spacing / 2)
# print(f"center_y_pos: {center_y_pos}")


def compute_angular_velocity(quat, prev_quat, dt):
    # Convert quaternions to scipy Rotation objects
    if prev_quat is None:
        prev_quat = quat
    r1 = R.from_quat(quat)  # Current quaternion
    r0 = R.from_quat(prev_quat)  # Previous quaternion

    # Compute relative rotation: r_rel = r0^(-1) * r1
    r_rel = r0.inv() * r1

    # Convert relative rotation to axis-angle
    axis, angle = r_rel.as_rotvec(), np.linalg.norm(r_rel.as_rotvec())

    # Angular velocity (in radians per second)
    angular_velocity = axis * (angle / dt)

    return list(angular_velocity)


# # convert to linear and angular velocity
def steps_to_vel(step_size, period):
    return (step_size * 2) / period


while True:
    pwe.tick(DT)
    if pwe.t <= 0 + args.skip_warmup * 1:
        start = pwe.t
        last_record = pwe.t + 1 / FPS
        last_meshcat_display = pwe.t + 1 / MESHCAT_FPS
        continue

    # print(np.around(pwe.robot.get_T_world_fbase()[:3, 3], 3))

    if pwe.t - last_record >= 1 / FPS:
        if args.stand:
            T_world_fbase = first_T_world_fbase
        else:
            T_world_fbase = pwe.robot.get_T_world_fbase()
        root_position = list(T_world_fbase[:3, 3])
        # if center_y_pos is None:
        #    center_y_pos = root_position[1]

        # TODO decomment this line for big duck ?
        # root_position[1] = root_position[1] - center_y_pos
        if round(root_position[2], 5) < 0:
            print(f"BAD root_position: {root_position[2]:.5f}")
        root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())
        joints_positions = list(pwe.get_angles().values())

        if args.stand:
            joints_positions = first_joints_positions
            T_world_leftFoot = first_T_world_leftFoot
            T_world_rightFoot = first_T_world_rightFoot
        else:
            joints_positions = list(pwe.get_angles().values())
            T_world_leftFoot = pwe.robot.get_T_world_left()
            T_world_rightFoot = pwe.robot.get_T_world_right()

        # T_body_leftFoot = (
        #     T_world_leftFoot  # np.linalg.inv(T_world_fbase) @ T_world_leftFoot
        # )
        # T_body_rightFoot = (
        #     T_world_rightFoot  # np.linalg.inv(T_world_fbase) @ T_world_rightFoot
        # )

        T_body_leftFoot = np.linalg.inv(T_world_fbase) @ T_world_leftFoot
        T_body_rightFoot = np.linalg.inv(T_world_fbase) @ T_world_rightFoot

        left_toe_pos = list(T_body_leftFoot[:3, 3])
        right_toe_pos = list(T_body_rightFoot[:3, 3])

        if not prev_initialized:
            prev_root_position = root_position.copy()
            prev_root_orientation_euler = (
                R.from_quat(root_orientation_quat).as_euler("xyz").copy()
            )
            prev_left_toe_pos = left_toe_pos.copy()
            prev_right_toe_pos = right_toe_pos.copy()
            prev_joints_positions = joints_positions.copy()
            prev_initialized = True

        world_linear_vel = list(
            (np.array(root_position) - np.array(prev_root_position)) / (1 / FPS)
        )
        avg_x_lin_vel.append(world_linear_vel[0])
        avg_y_lin_vel.append(world_linear_vel[1])
        body_rot_mat = T_world_fbase[:3, :3]
        body_linear_vel = list(body_rot_mat.T @ world_linear_vel)
        # print("world linear vel", world_linear_vel)
        # print("body linear vel", body_linear_vel)

        world_angular_vel = compute_angular_velocity(
            root_orientation_quat, prev_root_orientation_quat, (1 / FPS)
        )

        # world_angular_vel = list(
        #     (
        #         R.from_quat(root_orientation_quat).as_euler("xyz")
        #         - prev_root_orientation_euler
        #     )
        #     / (1 / FPS)
        # )
        avg_yaw_vel.append(world_angular_vel[2])
        body_angular_vel = list(body_rot_mat.T @ world_angular_vel)
        # print("world angular vel", world_angular_vel)
        # print("body angular vel", body_angular_vel)

        if prev_joints_positions == None:
            prev_joints_positions = [0] * len(joints_positions)
        joints_vel = list(
            (np.array(joints_positions) - np.array(prev_joints_positions)) / (1 / FPS)
        )
        left_toe_vel = list(
            (np.array(left_toe_pos) - np.array(prev_left_toe_pos)) / (1 / FPS)
        )
        right_toe_vel = list(
            (np.array(right_toe_pos) - np.array(prev_right_toe_pos)) / (1 / FPS)
        )

        foot_contacts = pwe.get_current_support_phase()

        if prev_initialized:
            if args.hardware:
                episode["Frames"].append(
                    root_position
                    + root_orientation_quat
                    + joints_positions
                    + left_toe_pos
                    + right_toe_pos
                    + world_linear_vel
                    + world_angular_vel
                    + joints_vel
                    + left_toe_vel
                    + right_toe_vel
                    + foot_contacts
                )
            else:
                episode["Frames"].append(
                    root_position + root_orientation_quat + joints_positions
                )
            if args.debug:
                episode["Debug_info"].append(
                    {
                        "left_foot_pose": list(T_world_leftFoot.flatten()),
                        "right_foot_pose": list(T_world_rightFoot.flatten()),
                    }
                )
            if not added_frame_info:
                added_frame_info = True
                offset = 0
                offset_root_pos = offset
                offset = offset + len(root_position)
                offset_root_quat = offset
                offset = offset + len(root_orientation_quat)
                offset_joints_pos = offset
                offset = offset + len(joints_positions)
                offset_left_toe_pos = offset
                offset = offset + len(left_toe_pos)
                offset_right_toe_pos = offset
                offset = offset + len(right_toe_pos)
                offset_world_linear_vel = offset
                offset = offset + len(world_linear_vel)
                offset_world_angular_vel = offset
                offset = offset + len(world_angular_vel)
                offset_joints_vel = offset
                offset = offset + len(joints_vel)
                offset_left_toe_vel = offset
                offset = offset + len(left_toe_vel)
                offset_right_toe_vel = offset
                offset = offset + len(right_toe_vel)
                offset_foot_contacts = offset
                offset = offset + len(foot_contacts)

                episode["Joints"] = list(pwe.get_angles().keys())
                episode["Frame_offset"].append(
                    {
                        "root_pos": offset_root_pos,
                        "root_quat": offset_root_quat,
                        "joints_pos": offset_joints_pos,
                        "left_toe_pos": offset_left_toe_pos,
                        "right_toe_pos": offset_right_toe_pos,
                        "world_linear_vel": offset_world_linear_vel,
                        "world_angular_vel": offset_world_angular_vel,
                        "joints_vel": offset_joints_vel,
                        "left_toe_vel": offset_left_toe_vel,
                        "right_toe_vel": offset_right_toe_vel,
                        "foot_contacts": offset_foot_contacts,
                    }
                )
                episode["Frame_size"].append(
                    {
                        "root_pos": len(root_position),
                        "root_quat": len(root_orientation_quat),
                        "joints_pos": len(joints_positions),
                        "left_toe_pos": len(left_toe_pos),
                        "right_toe_pos": len(right_toe_pos),
                        "world_linear_vel": len(world_linear_vel),
                        "world_angular_vel": len(world_angular_vel),
                        "joints_vel": len(joints_vel),
                        "left_toe_vel": len(left_toe_vel),
                        "right_toe_vel": len(right_toe_vel),
                        "foot_contacts": len(foot_contacts),
                    }
                )

        prev_root_position = root_position.copy()
        prev_root_orientation_quat = root_orientation_quat.copy()
        prev_root_orientation_euler = (
            R.from_quat(root_orientation_quat).as_euler("xyz").copy()
        )
        prev_left_toe_pos = left_toe_pos.copy()
        prev_right_toe_pos = right_toe_pos.copy()
        prev_joints_positions = joints_positions.copy()
        prev_initialized = True

        last_record = pwe.t

    if DISPLAY_MESHCAT and pwe.t - last_meshcat_display >= 1 / MESHCAT_FPS:
        last_meshcat_display = pwe.t
        viz.display(pwe.robot.state.q)
        footsteps_viz(pwe.trajectory.get_supports())
        robot_frame_viz(pwe.robot, "trunk")
        robot_frame_viz(pwe.robot, "left_foot")
        robot_frame_viz(pwe.robot, "right_foot")

    # if pwe.t - start > args.length:
    #    break
    if len(episode["Frames"]) == args.length * FPS:
        break

    i += 1

# skip first 2 seconds to get better average speed
mean_avg_x_lin_vel = np.around(np.mean(avg_x_lin_vel[240:]), 4)
mean_avg_y_lin_vel = np.around(np.mean(avg_y_lin_vel[240:]), 4)
mean_yaw_vel = np.around(np.mean(avg_yaw_vel[240:]), 4)

print("recorded", len(episode["Frames"]), "frames")
print(f"avg lin_vel_x {mean_avg_x_lin_vel}")
print(f"avg lin_vel_y {mean_avg_y_lin_vel}")
print(f"avg yaw {mean_yaw_vel}")
episode["Vel_x"] = mean_avg_x_lin_vel
episode["Vel_y"] = mean_avg_y_lin_vel
episode["Yaw"] = mean_yaw_vel
episode["Placo"] = {
    "dx": args.dx,
    "dy": args.dy,
    "dtheta": args.dtheta,
    "duration": args.length,
    "hardware": args.hardware,
    "double_support_ratio": pwe.parameters.double_support_ratio,
    "startend_double_support_ratio": pwe.parameters.startend_double_support_ratio,
    "planned_timesteps": pwe.parameters.planned_timesteps,
    "replan_timesteps": pwe.parameters.replan_timesteps,
    "walk_com_height": pwe.parameters.walk_com_height,
    "walk_foot_height": pwe.parameters.walk_foot_height,
    "walk_trunk_pitch": np.rad2deg(pwe.parameters.walk_trunk_pitch),
    "walk_foot_rise_ratio": pwe.parameters.walk_foot_rise_ratio,
    "single_support_duration": pwe.parameters.single_support_duration,
    "single_support_timesteps": pwe.parameters.single_support_timesteps,
    "foot_length": pwe.parameters.foot_length,
    "feet_spacing": pwe.parameters.feet_spacing,
    "zmp_margin": pwe.parameters.zmp_margin,
    "foot_zmp_target_x": pwe.parameters.foot_zmp_target_x,
    "foot_zmp_target_y": pwe.parameters.foot_zmp_target_y,
    "walk_max_dtheta": pwe.parameters.walk_max_dtheta,
    "walk_max_dy": pwe.parameters.walk_max_dy,
    "walk_max_dx_forward": pwe.parameters.walk_max_dx_forward,
    "walk_max_dx_backward": pwe.parameters.walk_max_dx_backward,
    "avg_x_lin_vel": mean_avg_x_lin_vel,
    "avg_y_lin_vel": mean_avg_y_lin_vel,
    "avg_yaw_vel": mean_yaw_vel,
    "preset_name": args.preset.split("/")[-1].split(".")[0],
    "period": pwe.period,
}


x_vel = np.around(steps_to_vel(args.dx, pwe.period), 3)
y_vel = np.around(steps_to_vel(args.dy, pwe.period), 3)
theta_vel = np.around(steps_to_vel(args.dtheta, pwe.period), 3)

print(f"computed xvel: {x_vel}, mean xvel: {mean_avg_x_lin_vel}")
print(f"computed yvel: {y_vel}, mean yvel: {mean_avg_y_lin_vel}")
print(f"computed thetavel: {theta_vel}, mean thetavel: {mean_yaw_vel}")


name = f"{args.name}_{x_vel}_{y_vel}_{theta_vel}" # Do we need the id in the name ?
# name = f"{x_vel}_{y_vel}_{theta_vel}"
file_name = name + str(".json")
file_path = os.path.join(args.output_dir, file_name)
os.makedirs(args.output_dir, exist_ok=True)
print("DONE, saving", file_name)
with open(file_path, "w") as f:
    json.encoder.c_make_encoder = None
    json.encoder.float = RoundingFloat
    json.dump(episode, f, indent=4)
