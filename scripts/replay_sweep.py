from FramesViewer.viewer import Viewer
import time
import argparse
from glob import glob
import os
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--directory", type=str, required=True, help="Directory containing the moves"
)
parser.add_argument("-k", action="store_true", default=False)
args = parser.parse_args()

COMMANDS_RANGE_X = [-0.5, 0.5]
COMMANDS_RANGE_Y = [-0.2, 0.2]
COMMANDS_RANGE_THETA = [-1.0, 1.0]


class ReferenceMotion:
    def __init__(self, directory):
        self.directory = directory
        # load all json files except metadata.json
        self.json_files = glob(self.directory + "/*.json")
        self.data = {}
        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.dx_range = [0, 0]
        self.dy_range = [0, 0]
        self.dtheta_range = [0, 0]
        self.dxs = []
        self.dys = []
        self.dthetas = []
        self.data_array = []
        self.process()
        # print(self.frame_offsets)
        # exit()

        self.slices = {}
        self.slices["root_pos"] = slice(
            self.frame_offsets["root_pos"], self.frame_offsets["root_quat"]
        )
        self.slices["root_quat"] = slice(
            self.frame_offsets["root_quat"], self.frame_offsets["joints_pos"]
        )
        self.slices["linear_vel"] = slice(
            self.frame_offsets["world_linear_vel"],
            self.frame_offsets["world_angular_vel"],
        )
        self.slices["angular_vel"] = slice(
            self.frame_offsets["world_angular_vel"], self.frame_offsets["joints_vel"]
        )
        self.slices["joints_pos"] = slice(
            self.frame_offsets["joints_pos"], self.frame_offsets["left_toe_pos"]
        )
        self.slices["joint_vels"] = slice(
            self.frame_offsets["joints_vel"], self.frame_offsets["left_toe_vel"]
        )
        self.slices["left_toe_pos"] = slice(
            self.frame_offsets["left_toe_pos"], self.frame_offsets["right_toe_pos"]
        )
        self.slices["right_toe_pos"] = slice(
            self.frame_offsets["right_toe_pos"], self.frame_offsets["world_linear_vel"]
        )


    def process(self):
        for file in self.json_files:

            if self.period is None:
                tmp_file = json.load(open(file))
                self.period = tmp_file["Placo"]["period"]
                self.fps = tmp_file["FPS"]
                self.frame_offsets = tmp_file["Frame_offset"][0]
                num_frames = len(tmp_file["Frames"])
                # print(num_frames)
                # exit()

            name = os.path.basename(file).strip(".json")
            split = name.split("_")
            id = float(split[0])
            dx = float(split[1])
            dy = float(split[2])
            dtheta = float(split[3])

            if dx not in self.dxs:
                self.dxs.append(dx)

            if dy not in self.dys:
                self.dys.append(dy)

            if dtheta not in self.dthetas:
                self.dthetas.append(dtheta)

            self.dx_range = [min(dx, self.dx_range[0]), max(dx, self.dx_range[1])]
            self.dy_range = [min(dy, self.dy_range[0]), max(dy, self.dy_range[1])]
            self.dtheta_range = [
                min(dtheta, self.dtheta_range[0]),
                max(dtheta, self.dtheta_range[1]),
            ]

            if dx not in self.data:
                self.data[dx]: dict = {}

            if dy not in self.data[dx]:
                self.data[dx][dy]: dict = {}

            if dtheta not in self.data[dx][dy]:
                self.data[dx][dy][dtheta] = json.load(open(file))["Frames"]

        self.dxs = sorted(self.dxs)
        self.dys = sorted(self.dys)
        self.dthetas = sorted(self.dthetas)

        print("dx range: ", self.dx_range)
        print("dy range: ", self.dy_range)
        print("dtheta range: ", self.dtheta_range)

        nb_dx = len(self.dxs)
        nb_dy = len(self.dys)
        nb_dtheta = len(self.dthetas)

        print("nb dx", nb_dx)
        print("nb dy", nb_dy)
        print("nb dtheta", nb_dtheta)

        self.data_array = nb_dx * [None]

        for x, dx in enumerate(self.dxs):
            self.data_array[x] = nb_dy * [None]
            for y, dy in enumerate(self.dys):
                self.data_array[x][y] = nb_dtheta * [None]
                for t, dtheta in enumerate(self.dthetas):
                    self.data_array[x][y][t] = self.data[dx][dy][dtheta]

        print(np.array(self.data_array).shape)
        # exit()

    def vel_to_index(self, dx, dy, dtheta):
        # First, clamp/cap to the ranges
        dx = min(max(dx, self.dx_range[0]), self.dx_range[1])
        dy = min(max(dy, self.dy_range[0]), self.dy_range[1])
        dtheta = min(max(dtheta, self.dtheta_range[0]), self.dtheta_range[1])

        # Find the closest values in self.dxs, self.dys, self.dthetas
        ix = np.argmin(np.abs(np.array(self.dxs) - dx))
        iy = np.argmin(np.abs(np.array(self.dys) - dy))
        itheta = np.argmin(np.abs(np.array(self.dthetas) - dtheta))

        return ix, iy, itheta

    # def get_closest_value(self, value, values):
    #     return min(values, key=lambda x: abs(x - value))

    # def get_closest_values(self, dx, dy, dtheta):
    #     # use vel_to_index to find the closest value in the data_array

    #     ix, iy, itheta = self.vel_to_index(dx, dy, dtheta)

    def get_closest_reference_motion(self, dx, dy, dtheta, i):
        ix, iy, itheta = self.vel_to_index(dx, dy, dtheta)
        print(ix, iy, itheta)
        return self.data_array[ix][iy][itheta][i]

        # dx = self.get_closest_value(dx, self.dxs)
        # dy = self.get_closest_value(dy, self.dys)
        # dtheta = self.get_closest_value(dtheta, self.dthetas)

        # return self.data[dx][dy][dtheta][i]


commands = [0.0, 0, 0]


def handle_keyboard():
    global commands
    keys = pygame.key.get_pressed()
    lin_vel_x = 0
    lin_vel_y = 0
    ang_vel = 0
    if keys[pygame.K_z]:
        lin_vel_x = COMMANDS_RANGE_X[1]
    if keys[pygame.K_s]:
        lin_vel_x = COMMANDS_RANGE_X[0]
    if keys[pygame.K_q]:
        lin_vel_y = COMMANDS_RANGE_Y[1]
    if keys[pygame.K_d]:
        lin_vel_y = COMMANDS_RANGE_Y[0]
    if keys[pygame.K_a]:
        ang_vel = COMMANDS_RANGE_THETA[1]
    if keys[pygame.K_e]:
        ang_vel = COMMANDS_RANGE_THETA[0]

    commands[0] = lin_vel_x
    commands[1] = lin_vel_y
    commands[2] = ang_vel

    # print(commands)

    pygame.event.pump()  # process event queue


if __name__ == "__main__":
    fv = Viewer()
    fv.start()

    if args.k:
        import pygame

        pygame.init()
        # open a blank pygame window
        screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Press arrow keys to move robot")

    RM = ReferenceMotion(args.directory)
    i = 0
    nb_steps_per_period = int(RM.fps * RM.period)  #  *10 # ??
    command = [-0.1, 0, 0]
    root_pose = np.eye(4)
    left_toe_pose = np.eye(4)
    right_toe_pose = np.eye(4)
    while True:

        if args.k:
            handle_keyboard()
        i += 1
        i = i % 1000
        ref = RM.get_closest_reference_motion(*commands, i)

        root_position = ref[RM.slices["root_pos"]]
        root_orientation_quat = ref[RM.slices["root_quat"]]
        root_orientation_mat = R.from_quat(root_orientation_quat).as_matrix()
        root_pose[:3, 3] = root_position
        root_pose[:3, :3] = root_orientation_mat

        left_toe_pos = np.array(ref[RM.slices["left_toe_pos"]]) + np.array(root_position)
        right_toe_pos = np.array(ref[RM.slices["right_toe_pos"]]) + np.array(root_position)

        left_toe_pose[:3, 3] = left_toe_pos
        right_toe_pose[:3, 3] = right_toe_pos

        fv.pushFrame(root_pose, "root")

        fv.pushFrame(left_toe_pose, "left_toe")
        fv.pushFrame(right_toe_pose, "right_toe")

        time.sleep(1 / RM.fps)
