import numpy as np
import json
from glob import glob
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--ref_motion", type=str, default="ref_motion")
args = parser.parse_args()

all_files = glob(f"{args.ref_motion}/*.json")


def fit_ref_motion(file):
    data = json.load(open(file))
    Y_all = np.array(data["Frames"])
    period = data["Placo"]["period"]
    fps = data["FPS"]
    frame_offsets = data["Frame_offset"][0]
    startend_double_support_ratio = data["Placo"]["startend_double_support_ratio"]
    start_offset = int(startend_double_support_ratio * fps)
    nb_steps_in_period = int(period * fps)
    _Y = Y_all[start_offset : start_offset + int(nb_steps_in_period)]
    joints_pos = _Y[:, frame_offsets["joints_pos"] : frame_offsets["left_toe_pos"]]
    joints_vel = _Y[:, frame_offsets["joints_vel"] : frame_offsets["left_toe_vel"]]
    foot_contacts = _Y[
        :, frame_offsets["foot_contacts"] : frame_offsets["foot_contacts"] + 2
    ]
    base_linear_vel = _Y[
        :, frame_offsets["world_linear_vel"] : frame_offsets["world_angular_vel"]
    ]
    base_angular_vel = _Y[
        :, frame_offsets["world_angular_vel"] : frame_offsets["joints_vel"]
    ]

    Y = np.concatenate(
        [joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel],
        axis=1,
    ).astype(np.float32)

    # Generate time feature
    X = np.linspace(0, 1, Y.shape[0]).reshape(-1, 1).astype(np.float32)  # Time variable

    # Polynomial degree
    degree = 15

    # Store coefficients
    coefficients = {}

    # ====== Fit Polynomial Regression per Dimension ======
    for dim in range(Y.shape[1]):
        coeffs = np.polyfit(X.flatten(), Y[:, dim], degree)
        coefficients[f"dim_{dim}"] = list(np.flip(coeffs))

    ret_data = {
        "coefficients": coefficients,
        "period": period,
        "fps": fps,
        "frame_offsets": frame_offsets,
        "start_offset": start_offset,
        "nb_steps_in_period": nb_steps_in_period,
        "startend_double_support_ratio": startend_double_support_ratio,
    }
    print(frame_offsets)
    exit()

    return ret_data


all_coefficients = {}
for file in all_files:
    name = os.path.basename(file).strip(".json")
    tmp = name.split("_")
    name = f"{tmp[1]}_{tmp[2]}_{tmp[3]}"

    all_coefficients[name] = fit_ref_motion(file)


pickle.dump(all_coefficients, open("polynomial_coefficients.pkl", "wb"))