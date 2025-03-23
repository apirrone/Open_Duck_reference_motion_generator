import os
import numpy as np
from glob import glob
import json
import matplotlib.pyplot as plt
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--coefficients",
    type=str,
    default="polynomial_coefficients.pkl",
    help="Path to polynomial coefficients file",
)
parser.add_argument("-n", type=int, default=1, help="number of periods to sample")
parser.add_argument("--ref_motion", type=str, default="recordings")
args = parser.parse_args()

# poly_data = json.load(open(args.coefficients))
poly_data = pickle.load(open(args.coefficients, "rb"))
# pickle.load(args.coefficients)

all_ref_files = glob(f"{args.ref_motion}/*.json")


def sample_polynomial(time: float, dimension: int, coefficients) -> float:
    """
    Evaluate the polynomial at a given time for a specific dimension.
    """
    if not (0 <= time <= 1):
        raise ValueError("Time must be between 0 and 1.")

    coeffs = coefficients.get(f"dim_{dimension}")
    if coeffs is None:
        raise ValueError(f"Dimension {dimension} not found in stored coefficients.")

    return sum(c * (time**i) for i, c in enumerate(coeffs))


for file in all_ref_files:
    name = os.path.basename(file).strip(".json")
    tmp = name.split("_")
    name = f"{tmp[1]}_{tmp[2]}_{tmp[3]}"
    banner = f"=== {name} ==="
    spacer = "=" * len(banner)
    print(spacer)
    print(banner)
    print(spacer)

    period = poly_data[name]["period"]
    fps = poly_data[name]["fps"]
    frame_offsets = poly_data[name]["frame_offsets"]
    startend_double_support_ratio = poly_data[name]["startend_double_support_ratio"]
    start_offset = int(startend_double_support_ratio * fps)
    nb_steps_in_period = int(period * fps)

    # Load reference motion data
    ref_data = json.load(open(file))
    Y_all = np.array(ref_data["Frames"])
    _Y = Y_all[start_offset : start_offset + int(nb_steps_in_period) * args.n]
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
    )

    # sigmaban
    dimensions_names = [
        "pos head_yaw",
        "pos head_pitch",
        "pos left_shoulder_pitch",
        "pos left_shoulder_roll",
        "pos left_elbow",
        "pos right_shoulder_pitch",
        "pos right_shoulder_roll",
        "pos right_elbow",
        "pos left_hip_yaw",
        "pos left_hip_roll",
        "pos left_hip_pitch",
        "pos left_knee",
        "pos left_ankle_pitch",
        "pos left_ankle_roll",
        "pos right_hip_yaw",
        "pos right_hip_roll",
        "pos right_hip_pitch",
        "pos right_knee",
        "pos right_ankle_pitch",
        "pos right_ankle_roll",

        "vel head_yaw",
        "vel head_pitch",
        "vel left_shoulder_pitch",
        "vel left_shoulder_roll",
        "vel left_elbow",
        "vel right_shoulder_pitch",
        "vel right_shoulder_roll",
        "vel right_elbow",
        "vel left_hip_yaw",
        "vel left_hip_roll",
        "vel left_hip_pitch",
        "vel left_knee",
        "vel left_ankle_pitch",
        "vel left_ankle_roll",
        "vel right_hip_yaw",
        "vel right_hip_roll",
        "vel right_hip_pitch",
        "vel right_knee",
        "vel right_ankle_pitch",
        "vel right_ankle_roll",

        "foot_contacts left",
        "foot_contacts right",

        "base_linear_vel x",
        "base_linear_vel y",
        "base_linear_vel z",

        "base_angular_vel x",
        "base_angular_vel y",
        "base_angular_vel z",
]


    # open duck mini
    # dimensions_names = [
    #     "pos left_hip_yaw",
    #     "pos left_hip_roll",
    #     "pos left_hip_pitch",
    #     "pos left_knee",
    #     "pos left_ankle",
    #     "pos neck_pitch",
    #     "pos head_pitch",
    #     "pos head_yaw",
    #     "pos head_roll",
    #     "pos left_antenna",
    #     "pos right_antenna",
    #     "pos right_hip_yaw",
    #     "pos right_hip_roll",
    #     "pos right_hip_pitch",
    #     "pos right_knee",
    #     "pos right_ankle",
    #     "vel left_hip_yaw",
    #     "vel left_hip_roll",
    #     "vel left_hip_pitch",
    #     "vel left_knee",
    #     "vel left_ankle",
    #     "vel neck_pitch",
    #     "vel head_pitch",
    #     "vel head_yaw",
    #     "vel head_roll",
    #     "vel left_antenna",
    #     "vel right_antenna",
    #     "vel right_hip_yaw",
    #     "vel right_hip_roll",
    #     "vel right_hip_pitch",
    #     "vel right_knee",
    #     "vel right_ankle",
    #     "foot_contacts left",
    #     "foot_contacts right",
    #     "base_linear_vel x",
    #     "base_linear_vel y",
    #     "base_linear_vel z",
    #     "base_angular_vel x",
    #     "base_angular_vel y",
    #     "base_angular_vel z",
    # ]
    nb_dim = Y.shape[1]

    # Generate time feature
    X = np.linspace(0, 1, Y.shape[0]).reshape(-1, 1)  # Time variable

    # Get coefficients for this motion
    coefficients = poly_data[name]["coefficients"]

    dimensions = []

    # ====== Plot Fitting for Each Dimension ======
    for dimension in range(nb_dim):
        poly_samples = []
        ref_samples = []
        # times = X.flatten()  # Use original time values for accurate mapping
        times = np.arange(len(Y)) % nb_steps_in_period / nb_steps_in_period

        for i, t in enumerate(times):
            if i >= Y.shape[0]:
                break  # Prevent out-of-bounds errors

            poly_sample = sample_polynomial(t, dimension, coefficients)
            ref = Y[i, dimension]

            poly_samples.append(poly_sample)
            ref_samples.append(ref)
        dimensions.append((dimension, ref_samples, poly_samples))

    # ====== Plotting ======
    nb_rows = int(np.sqrt(nb_dim))
    nb_cols = int(np.ceil(nb_dim / nb_rows))

    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(12, 8))
    axs = axs.flatten()

    for i, (dim, ref, poly) in enumerate(dimensions):
        ax = axs[i]
        ax.plot(ref, label="Original Data", alpha=0.5)
        ax.plot(poly, label="Polynomial Fit", color="red")
        ax.set_title(f"{dimensions_names[dim]}")

    plt.tight_layout()
    plt.show()
