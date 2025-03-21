import argparse
from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import webbrowser
import threading
import json
import os
import time
from os.path import join
from threading import current_thread

import numpy as np
import placo
from placo_utils.visualization import footsteps_viz, robot_frame_viz, robot_viz, get_viewer
from scipy.spatial.transform import Rotation as R

from placo_walk_engine import PlacoWalkEngine

parser = argparse.ArgumentParser()
parser.add_argument("--dx", type=float, default=0)
parser.add_argument("--dy", type=float, default=0)
parser.add_argument("--output_dir", type=str, default=".")
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
parser.add_argument(
    "--duck",
    choices=["go_bdx", "open_duck_mini", "open_duck_mini_v2", "sigmaban2019"],
    help="Duck type",
    required=True,
)
parser.add_argument("--preset", type=str, help="Path to the preset file")
args = parser.parse_args()

app = Flask(__name__)

FPS = 60
MESHCAT_FPS = 60
DT = 0.001

episode = {
    "LoopMode": "Wrap",
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Debug_info": [],
    "Frames": [],
    "MotionWeight": 1,
}

def open_browser():
    try:
        webbrowser.open_new('http://127.0.0.1:5000/')
    except:
        print("Failed to open the default browser. Trying Google Chrome.")
        try:
            webbrowser.get('google-chrome').open_new('http://127.0.0.1:5000/')
        except:
            print("Failed to open Google Chrome. Make sure it's installed and accessible.")

script_path = os.path.dirname(os.path.abspath(__file__))
# Define the parameters class to hold the variables
class GaitParameters:
    def __init__(self):
        if args.duck == "open_duck_mini":
            self.robot = 'open_duck_mini'
            self.robot_urdf = "open_duck_mini.urdf"
            self.asset_path = os.path.join(script_path, "../open_duck_reference_motion_generator/robots/open_duck_mini/")
        elif args.duck == "open_duck_mini_v2":
            self.robot = 'open_duck_mini_v2'
            self.robot_urdf = "open_duck_mini_v2.urdf"
            self.asset_path = os.path.join(script_path, "../open_duck_reference_motion_generator/robots/open_duck_mini_v2/")
        elif args.duck == "go_bdx":
            self.robot = 'go_bdx'
            self.robot_urdf = "go_bdx.urdf"
            self.asset_path = os.path.join(script_path, "../open_duck_reference_motion_generator/robots/go_bdx/")
        elif args.duck == "sigmaban2019":
            self.robot = 'sigmaban2019'
            self.robot_urdf = "sigmaban2019.urdf"
            self.asset_path = os.path.join(script_path, "../open_duck_reference_motion_generator/robots/sigmaban2019/")
        self.dx = 0.1
        self.dy = 0.0
        self.dtheta = 0.0
        self.duration = 5
        self.hardware = True

    def reset(self, pwe):
        pwe.parameters.double_support_ratio = self.double_support_ratio
        pwe.parameters.startend_double_support_ratio = self.startend_double_support_ratio
        pwe.parameters.planned_timesteps = self.planned_timesteps
        pwe.parameters.replan_timesteps = self.replan_timesteps
        pwe.parameters.walk_com_height = self.walk_com_height
        pwe.parameters.walk_foot_height = self.walk_foot_height
        pwe.parameters.walk_trunk_pitch = np.deg2rad(self.walk_trunk_pitch)
        pwe.parameters.walk_foot_rise_ratio = self.walk_foot_rise_ratio
        pwe.parameters.single_support_duration = self.single_support_duration
        pwe.parameters.single_support_timesteps = self.single_support_timesteps
        pwe.parameters.foot_length = self.foot_length
        pwe.parameters.feet_spacing = self.feet_spacing
        pwe.parameters.zmp_margin = self.zmp_margin
        pwe.parameters.foot_zmp_target_x = self.foot_zmp_target_x
        pwe.parameters.foot_zmp_target_y = self.foot_zmp_target_y
        pwe.parameters.walk_max_dtheta = self.walk_max_dtheta
        pwe.parameters.walk_max_dy = self.walk_max_dy
        pwe.parameters.walk_max_dx_forward = self.walk_max_dx_forward
        pwe.parameters.walk_max_dx_backward = self.walk_max_dx_backward

    def save_to_json(self, filename):
        data = {
            'dx': self.dx,
            'dy': self.dy,
            'dtheta': self.dtheta,
            'duration': self.duration,
            'hardware': self.hardware,
            'double_support_ratio': self.double_support_ratio,
            'startend_double_support_ratio': self.startend_double_support_ratio,
            'planned_timesteps': self.planned_timesteps,
            'replan_timesteps': self.replan_timesteps,
            'walk_com_height': self.walk_com_height,
            'walk_foot_height': self.walk_foot_height,
            'walk_trunk_pitch': self.walk_trunk_pitch,
            'walk_foot_rise_ratio': self.walk_foot_rise_ratio,
            'single_support_duration': self.single_support_duration,
            'single_support_timesteps': self.single_support_timesteps,
            'foot_length': self.foot_length,
            'feet_spacing': self.feet_spacing,
            'zmp_margin': self.zmp_margin,
            'foot_zmp_target_x': self.foot_zmp_target_x,
            'foot_zmp_target_y': self.foot_zmp_target_y,
            'walk_max_dtheta': self.walk_max_dtheta,
            'walk_max_dy': self.walk_max_dy,
            'walk_max_dx_forward': self.walk_max_dx_forward,
            'walk_max_dx_backward': self.walk_max_dx_backward,
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def create_pwe(self, parameters=None):
        pwe = PlacoWalkEngine(self.asset_path, self.robot_urdf, parameters)
        self.reset(pwe)
        pwe.set_traj(0, 0, 0)
        return pwe

    def custom_preset_name(self):
        return f"placo_{gait.robot}_defaults.json"

    def save_custom_presets(self):
        filename = self.custom_preset_name()
        self.save_to_json(filename)

    def load_defaults(self, pwe):
        self.load_from_json(os.path.join(pwe.asset_path, "placo_defaults.json"))

    def load_from_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.load_from_data(data)

    def load_from_data(self, data):
        self.dx = data.get('dx')
        self.dy = data.get('dy')
        self.dtheta = data.get('dtheta')
        self.duration = data.get('duration')
        self.hardware = data.get('hardware')
        self.double_support_ratio = data.get('double_support_ratio')
        self.startend_double_support_ratio = data.get('startend_double_support_ratio')
        self.planned_timesteps = data.get('planned_timesteps')
        self.replan_timesteps = data.get('replan_timesteps')
        self.walk_com_height = data.get('walk_com_height')
        self.walk_foot_height = data.get('walk_foot_height')
        self.walk_trunk_pitch = data.get('walk_trunk_pitch')
        self.walk_foot_rise_ratio = data.get('walk_foot_rise_ratio')
        self.single_support_duration = data.get('single_support_duration')
        self.single_support_timesteps = data.get('single_support_timesteps')
        self.foot_length = data.get('foot_length')
        self.feet_spacing = data.get('feet_spacing')
        self.zmp_margin = data.get('zmp_margin')
        self.foot_zmp_target_x = data.get('foot_zmp_target_x')
        self.foot_zmp_target_y = data.get('foot_zmp_target_y')
        self.walk_max_dtheta = data.get('walk_max_dtheta')
        self.walk_max_dy = data.get('walk_max_dy')
        self.walk_max_dx_forward = data.get('walk_max_dx_forward')
        self.walk_max_dx_backward = data.get('walk_max_dx_backward')

if __name__ == '__main__':
    print('exit')

gait = GaitParameters()
if args.preset:
    filename = args.preset
else:
    filename = gait.custom_preset_name()
    if not os.path.exists(filename):
        filename = os.path.join(gait.asset_path, "placo_defaults.json")
with open(filename, 'r') as f:
    gait_parameters = json.load(f)
    print(f"gait_parameters {gait_parameters}")
    if args.double_support_ratio is not None:
        gait_parameters['double_support_ratio'] = args.double_support_ratio
    if args.startend_double_support_ratio is not None:
        gait_parameters['startend_double_support_ratio'] = args.startend_double_support_ratio
    if args.planned_timesteps is not None:
        gait_parameters['planned_timesteps'] = args.planned_timesteps
    if args.replan_timesteps is not None:
        gait_parameters['replan_timesteps'] = args.replan_timesteps
    if args.walk_com_height is not None:
        gait_parameters['walk_com_height'] = args.walk_com_height
    if args.walk_foot_height is not None:
        gait_parameters['walk_foot_height'] = args.walk_foot_height
    if args.walk_trunk_pitch is not None:
        gait_parameters['walk_trunk_pitch'] = np.deg2rad(args.walk_trunk_pitch)
    if args.walk_foot_rise_ratio is not None:
        gait_parameters['walk_foot_rise_ratio'] = args.walk_foot_rise_ratio
    if args.single_support_duration is not None:
        gait_parameters['single_support_duration'] = args.single_support_duration
    if args.single_support_timesteps is not None:
        gait_parameters['single_support_timesteps'] = args.single_support_timesteps
    if args.foot_length is not None:
        gait_parameters['foot_length'] = args.foot_length
    if args.feet_spacing is not None:
        gait_parameters['feet_spacing'] = args.feet_spacing
    if args.zmp_margin is not None:
        gait_parameters['zmp_margin'] = args.zmp_margin
    if args.foot_zmp_target_x is not None:
        gait_parameters['foot_zmp_target_x'] = args.foot_zmp_target_x
    if args.foot_zmp_target_y is not None:
        gait_parameters['foot_zmp_target_y'] = args.foot_zmp_target_y
    if args.walk_max_dtheta is not None:
        gait_parameters['walk_max_dtheta'] = args.walk_max_dtheta
    if args.walk_max_dy is not None:
        gait_parameters['walk_max_dy'] = args.walk_max_dy
    if args.walk_max_dx_forward is not None:
        gait_parameters['walk_max_dx_forward'] = args.walk_max_dx_forward
    if args.walk_max_dx_backward is not None:
        gait_parameters['walk_max_dx_backward'] = args.walk_max_dx_backward
    gait.load_from_data(gait_parameters)
pwe = gait.create_pwe(gait_parameters)
viz = robot_viz(pwe.robot)
viz.display(pwe.robot.state.q)

threading.Timer(1, open_browser).start()

run_loop = False
dorun = False
doreset = False
doupdate = False
gait_condition = threading.Condition()
# gait_start_semaphore = threading.Semaphore(0)

@app.route('/log', methods=['POST'])
def log_message():
    message = request.json.get('message', '')
    print(f"Placo: {message}")
    return 'Logged', 200

@app.route('/', methods=['GET', 'POST'])
def index():
    global run_loop
    return render_template('index.html', parameters=gait)

@app.route('/save_state', methods=['POST'])
def save_state():
    gait.save_custom_presets()
    return "", 200

@app.route('/defaults', methods=['GET'])
def defaults():
    if os.path.exists("bdx_state.json"):
        gait.load_from_json()
    else:
        gait.load_defaults(pwe)
    parameters = gait.__dict__
    return jsonify(parameters)

@app.route('/get', methods=['GET'])
def get_parameters():
    # Convert gait parameters to a dictionary
    parameters = gait.__dict__
    return jsonify(parameters)

@app.route('/set_playback_speed', methods=['POST'])
def set_playback_speed():
    global DT
    data = request.get_json()
    speed = data.get('speed')
    if speed == 0.25:
        DT = 0.00001
        return "Speed changed successfully", 200
    elif speed == 0.5:
        DT = 0.0001
        return "Speed changed successfully", 200
    elif speed == 1.0:
        DT = 0.001
        return "Speed changed successfully", 200
    return "Invalid speed selection", 400

@app.route('/change_robot', methods=['POST'])
def change_robot():
    global run_loop
    global doreset
    global dorun
    data = request.get_json()
    selected_robot = data.get('robot')
    print(f"selected_robot: {selected_robot}")
    if selected_robot in ['go_bdx', 'open_duck_mini']:
        if selected_robot != gait.robot:
            gait.robot = selected_robot
            # Reset the gait generator to use the new robot
            with gait_condition:
                run_loop = False
                doreset = True
                dorun = False
                gait_condition.notify()
        return "Robot changed successfully", 200
    else:
        return "Invalid robot selection", 400

@app.route('/run', methods=['POST'])
def run():
    global dorun
    global run_loop
    # Update parameters from sliders
    gait.dx = float(request.form['dx'])
    gait.dy = float(request.form['dy'])
    gait.dtheta = float(request.form['dtheta'])
    gait.duration = int(request.form['duration'])
    gait.double_support_ratio = float(request.form['double_support_ratio'])
    gait.startend_double_support_ratio = float(request.form['startend_double_support_ratio'])
    gait.planned_timesteps = int(request.form['planned_timesteps'])
    gait.replan_timesteps = int(request.form['replan_timesteps'])
    gait.walk_com_height = float(request.form['walk_com_height'])
    gait.walk_foot_height = float(request.form['walk_foot_height'])
    gait.walk_trunk_pitch = int(request.form['walk_trunk_pitch'])  # Degrees
    gait.walk_foot_rise_ratio = float(request.form['walk_foot_rise_ratio'])
    gait.single_support_duration = float(request.form['single_support_duration'])
    gait.single_support_timesteps = int(request.form['single_support_timesteps'])
    gait.foot_length = float(request.form['foot_length'])
    gait.feet_spacing = float(request.form['feet_spacing'])
    gait.zmp_margin = float(request.form['zmp_margin'])
    gait.foot_zmp_target_x = float(request.form['foot_zmp_target_x'])
    gait.foot_zmp_target_y = float(request.form['foot_zmp_target_y'])
    gait.walk_max_dtheta = float(request.form['walk_max_dtheta'])
    gait.walk_max_dy = float(request.form['walk_max_dy'])
    gait.walk_max_dx_forward = float(request.form['walk_max_dx_forward'])
    gait.walk_max_dx_backward = float(request.form['walk_max_dx_backward'])
    with gait_condition:
        run_loop = False
        dorun = True
        gait_condition.notify()
    return "", 200

@app.route('/update', methods=['POST'])
def update():
    global dorun
    global run_loop
    # Update parameters from sliders
    gait.dx = float(request.form['dx'])
    gait.dy = float(request.form['dy'])
    gait.dtheta = float(request.form['dtheta'])
    gait.duration = int(request.form['duration'])
    gait.double_support_ratio = float(request.form['double_support_ratio'])
    gait.startend_double_support_ratio = float(request.form['startend_double_support_ratio'])
    gait.planned_timesteps = int(request.form['planned_timesteps'])
    gait.replan_timesteps = int(request.form['replan_timesteps'])
    gait.walk_com_height = float(request.form['walk_com_height'])
    gait.walk_foot_height = float(request.form['walk_foot_height'])
    gait.walk_trunk_pitch = int(request.form['walk_trunk_pitch'])  # Degrees
    gait.walk_foot_rise_ratio = float(request.form['walk_foot_rise_ratio'])
    gait.single_support_duration = float(request.form['single_support_duration'])
    gait.single_support_timesteps = int(request.form['single_support_timesteps'])
    gait.foot_length = float(request.form['foot_length'])
    gait.feet_spacing = float(request.form['feet_spacing'])
    gait.zmp_margin = float(request.form['zmp_margin'])
    gait.foot_zmp_target_x = float(request.form['foot_zmp_target_x'])
    gait.foot_zmp_target_y = float(request.form['foot_zmp_target_y'])
    gait.walk_max_dtheta = float(request.form['walk_max_dtheta'])
    gait.walk_max_dy = float(request.form['walk_max_dy'])
    gait.walk_max_dx_forward = float(request.form['walk_max_dx_forward'])
    gait.walk_max_dx_backward = float(request.form['walk_max_dx_backward'])
    with gait_condition:
        dorun = run_loop
        run_loop = False
        gait_condition.notify()
    return "", 200

@app.route('/stop', methods=['POST'])
def stop():
    global run_loop
    global dorun
    dorun = False
    run_loop = False
    print("Stopping")
    return "Loop stopped successfully", 200

@app.route('/reset', methods=['POST'])
def reset():
    global run_loop
    global doreset
    with gait_condition:
        run_loop = False
        doreset = True
        gait_condition.notify()
    print("Resetting")
    return "Loop stopped successfully", 200

def gait_generator_thread():
    global run_loop
    global doreset
    global dorun
    global pwe
    global viz
    global DT
    while True:
        print("gait generator waiting")
        with gait_condition:
            if doreset:
                pwe = gait.create_pwe()
                viz = robot_viz(pwe.robot)
                viz.display(pwe.robot.state.q)
                footsteps_viz(pwe.trajectory.get_supports())
                robot_frame_viz(pwe.robot, "trunk")
                robot_frame_viz(pwe.robot, "left_foot")
                robot_frame_viz(pwe.robot, "right_foot")
                doreset = False
                continue
            if not dorun:
                gait_condition.wait()
                dorun = False
            if doreset:
                print("RESETTING")
                pwe = gait.create_pwe()
                viz = robot_viz(pwe.robot)
                viz.display(pwe.robot.state.q)
                footsteps_viz(pwe.trajectory.get_supports())
                robot_frame_viz(pwe.robot, "trunk")
                robot_frame_viz(pwe.robot, "left_foot")
                robot_frame_viz(pwe.robot, "right_foot")
                doreset = False
                continue
            run_loop = True
        gait.reset(pwe)
        pwe.set_traj(gait.dx, gait.dy, gait.dtheta + 0.001)
        start = pwe.t

        last_record = 0
        last_meshcat_display = 0
        prev_root_position = [0, 0, 0]
        prev_root_orientation_euler = [0, 0, 0]
        prev_left_toe_pos = [0, 0, 0]
        prev_right_toe_pos = [0, 0, 0]
        prev_joints_positions = None
        i = 0
        prev_initialized = False
        while run_loop:
            pwe.tick(DT)
            if pwe.t <= 0:
                # print("waiting ")
                start = pwe.t
                last_record = pwe.t + 1 / FPS
                last_meshcat_display = pwe.t + 1 / MESHCAT_FPS
                continue

            # print(np.around(pwe.robot.get_T_world_fbase()[:3, 3], 3))

            if pwe.t - last_record >= 1 / FPS:
                # before
                # T_world_fbase = pwe.robot.get_T_world_fbase()
                # after
                T_world_fbase = pwe.robot.get_T_world_trunk()
                # fv.pushFrame(T_world_fbase, "trunk")
                root_position = list(T_world_fbase[:3, 3])
                root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())
                joints_positions = list(pwe.get_angles().values())

                T_world_leftFoot = pwe.robot.get_T_world_left()
                T_world_rightFoot = pwe.robot.get_T_world_right()

                # fv.pushFrame(T_world_leftFoot, "left")
                # fv.pushFrame(T_world_rightFoot, "right")

                T_body_leftFoot = np.linalg.inv(T_world_fbase) @ T_world_leftFoot
                T_body_rightFoot = np.linalg.inv(T_world_fbase) @ T_world_rightFoot

                # left_foot_pose = pwe.robot.get_T_world_left()
                # right_foot_pose = pwe.robot.get_T_world_right()

                left_toe_pos = list(T_body_leftFoot[:3, 3])
                right_toe_pos = list(T_body_rightFoot[:3, 3])

                world_linear_vel = list(
                    (np.array(root_position) - np.array(prev_root_position)) / (1 / FPS)
                )
                body_rot_mat = T_world_fbase[:3, :3]
                body_linear_vel = list(body_rot_mat.T @ world_linear_vel)
                # print("world linear vel", world_linear_vel)
                # print("body linear vel", body_linear_vel)

                world_angular_vel = list(
                    (
                        R.from_quat(root_orientation_quat).as_euler("xyz")
                        - prev_root_orientation_euler
                    )
                    / (1 / FPS)
                )
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
                    if gait.hardware:
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
                    episode["Debug_info"].append(
                        {
                            "left_foot_pose": list(T_world_leftFoot.flatten()),
                            "right_foot_pose": list(T_world_rightFoot.flatten()),
                        }
                    )

                prev_root_position = root_position.copy()
                prev_root_orientation_euler = (
                    R.from_quat(root_orientation_quat).as_euler("xyz").copy()
                )
                prev_left_toe_pos = left_toe_pos.copy()
                prev_right_toe_pos = right_toe_pos.copy()
                prev_joints_positions = joints_positions.copy()
                prev_initialized = True

                last_record = pwe.t
                # print("saved frame")

            if pwe.t - last_meshcat_display >= 1 / MESHCAT_FPS:
                last_meshcat_display = pwe.t
                viz.display(pwe.robot.state.q)
                footsteps_viz(pwe.trajectory.get_supports())
                robot_frame_viz(pwe.robot, "trunk")
                robot_frame_viz(pwe.robot, "left_foot")
                robot_frame_viz(pwe.robot, "right_foot")

            if pwe.t - start > gait.duration:
                break

            i += 1
        run_loop = False
        # print("recorded", len(episode["Frames"]), "frames")
        # args_name = "dummy"
        # file_name = args_name + str(".txt")
        # file_path = os.path.join(args.output_dir, file_name)
        # os.makedirs(args.output_dir, exist_ok=True)
        # print("DONE, saving", file_name)
        # with open(file_path, "w") as f:
        #     json.dump(episode, f)

def open_browser():
    try:
        webbrowser.open_new('http://127.0.0.1:5000/')
    except:
        print("Failed to open the default browser. Trying Google Chrome.")
        try:
            webbrowser.get('google-chrome').open_new('http://127.0.0.1:5000/')
        except:
            print("Failed to open Google Chrome. Make sure it's installed and accessible.")

thread = threading.Thread(target=gait_generator_thread, daemon=True)
thread.start()

if __name__ == '__main__':
    app.run(debug=False)
