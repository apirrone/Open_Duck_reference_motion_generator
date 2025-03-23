import argparse
import json
import time

import FramesViewer.utils as fv_utils
import numpy as np
from FramesViewer.viewer import Viewer
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
parser.add_argument(
    "--hardware",
    action="store_true",
    help="use AMP_for_hardware format. If false, use IsaacGymEnvs format",
)
args = parser.parse_args()

fv = Viewer()
fv.start()

episode = json.load(open(args.file))

frame_duration = episode["FrameDuration"]

frames = episode["Frames"]
frame_offsets = episode["Frame_offset"][0]

root_pos_slice = slice(frame_offsets["root_pos"], frame_offsets["root_quat"])
root_quat_slice = slice(frame_offsets["root_quat"], frame_offsets["joints_pos"])

linear_vel_slice = slice(frame_offsets["world_linear_vel"], frame_offsets["world_angular_vel"])
angular_vel_slice = slice(frame_offsets["world_angular_vel"], frame_offsets["joints_vel"])
joint_vels_slice = slice(frame_offsets["joints_vel"], frame_offsets["left_toe_vel"])

left_toe_pos_slice = slice(frame_offsets["left_toe_pos"], frame_offsets["right_toe_pos"])
right_toe_pos_slice = slice(frame_offsets["right_toe_pos"], frame_offsets["world_linear_vel"])

if "Debug_info" in episode:
    debug = episode["Debug_info"]
else:
    debug = None
pose = np.eye(4)
vels = {}
vels["linear_vel"] = []
vels["angular_vel"] = []
vels["joint_vels"] = []
for i, frame in enumerate(frames):
    root_position = frame[root_pos_slice]
    root_orientation_quat = frame[root_quat_slice]
    root_orientation_mat = R.from_quat(root_orientation_quat).as_matrix()

    pose[:3, 3] = root_position
    pose[:3, :3] = root_orientation_mat

    fv.pushFrame(pose, "aze")

    vels["linear_vel"].append(frame[linear_vel_slice])
    vels["angular_vel"].append(frame[angular_vel_slice])
    vels["joint_vels"].append(frame[joint_vels_slice])

    left_toe_pos = np.array(frame[left_toe_pos_slice]) + np.array(root_position)
    right_toe_pos = np.array(frame[right_toe_pos_slice]) + np.array(root_position)
    
    fv.pushFrame(fv_utils.make_pose(left_toe_pos, [0, 0, 0]), "left_toe")
    fv.pushFrame(fv_utils.make_pose(right_toe_pos, [0, 0, 0]), "right_toe")

    time.sleep(frame_duration)


# plot vels
x_lin_vel = [vels["linear_vel"][i][0] for i in range(len(frames))]
y_lin_vel = [vels["linear_vel"][i][1] for i in range(len(frames))]
z_lin_vel = [vels["linear_vel"][i][2] for i in range(len(frames))]

joints_vel = [vels["joint_vels"][i] for i in range(len(frames))]
angular_vel_x = [vels["angular_vel"][i][0] for i in range(len(frames))]
angular_vel_y = [vels["angular_vel"][i][1] for i in range(len(frames))]
angular_vel_z = [vels["angular_vel"][i][2] for i in range(len(frames))]

plt.plot(angular_vel_x, label="angular_vel_x")
plt.plot(angular_vel_y, label="angular_vel_y")
plt.plot(angular_vel_z, label="angular_vel_z")

plt.legend()
plt.show()
