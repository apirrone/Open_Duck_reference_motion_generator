import os
import re
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


def run_command_with_logging(cmd_log_tuple):
    cmd, log_file = cmd_log_tuple
    if log_file is None:
        print(f"cmd: {cmd}")
        subprocess.run(cmd)
    else:
        with open(log_file, "w") as outfile:
            subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT)


def numeric_prefix_sort_key(item):
    total_speed, preset_name = item
    match = re.match(r"(\d+)(.*)", preset_name)
    if match:
        number_part = int(match.group(1))
        rest_part = match.group(2)
        return (number_part, rest_part)
    return (float("inf"), preset_name)


def main(args):
    """
    Generates random preset data, creates gait motions, filters recordings,
    and (optionally) plots anim/sim if --plot is given.
    """

    start_time = time.time()

    # ---------------------------------------------------------------
    # 1. Load parameters from auto_gait.json
    # ---------------------------------------------------------------
    # This JSON should contain keys like: slow, medium, fast, dx_max, dy_max, dtheta_max
    props_path = f"{SCRIPT_PATH}/../open_duck_reference_motion_generator/robots/{args.duck}/auto_gait.json"
    if not os.path.isfile(props_path):
        raise FileNotFoundError(f"Could not find props file at: {props_path}")

    with open(props_path, "r") as f:
        gait_props = json.load(f)

    # Extract needed values
    slow = gait_props["slow"]
    medium = gait_props["medium"]
    fast = gait_props["fast"]
    dx_max = gait_props["dx_max"]  # e.g. [0, 0.05]
    dy_max = gait_props["dy_max"]  # e.g. [0, 0.05]
    dtheta_max = gait_props["dtheta_max"]  # e.g. [0, 0.25]
    min_sweep_x = gait_props["min_sweep_x"]
    max_sweep_x = gait_props["max_sweep_x"]
    min_sweep_y = gait_props["min_sweep_y"]
    max_sweep_y = gait_props["max_sweep_y"]
    min_sweep_theta = gait_props["min_sweep_theta"]
    max_sweep_theta = gait_props["max_sweep_theta"]
    sweep_xy_granularity = gait_props["sweep_xy_granularity"]
    sweep_theta_granularity = gait_props["sweep_theta_granularity"]

    # For convenience, keep these in a dict for the gait motions
    gait_speeds = {"slow": slow, "medium": medium, "fast": fast}

    # Define the typical gait motions we want to generate
    gait_motions = [
        "standing",
        "forward",
        "backward",
        "left",
        "right",
        "ang_left",
        "ang_right",
        "dia_forward",
        "dia_backward",
    ]

    # ---------------------------------------------------------------
    # 2. Paths and directories
    # ---------------------------------------------------------------
    presets_dir = f"{SCRIPT_PATH}/../open_duck_reference_motion_generator/robots/{args.duck}/placo_presets"
    tmp_dir = os.path.join(presets_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    log_dir = os.path.join(args.output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 3. Generate random presets (n times)
    #    "medium" and "fast" as example base speeds
    # ---------------------------------------------------------------
    preset_speeds = ["medium"]  # , "fast"]

    if args.sweep:
        dxs = np.arange(min_sweep_x, max_sweep_x + sweep_xy_granularity, sweep_xy_granularity)
        dys = np.arange(min_sweep_y, max_sweep_y + sweep_xy_granularity, sweep_xy_granularity)
        dthetas = np.arange(min_sweep_theta, max_sweep_theta + sweep_theta_granularity, sweep_theta_granularity)
        all_n = len(dxs) * len(dys) * len(dthetas)
    else:
        all_n = args.num

    nb_moves_message = f"=== GENERATING {all_n} MOVES ==="
    spacer = "=" * len(nb_moves_message)
    print(spacer)
    print(nb_moves_message)
    print(spacer)

    commands = []
    for i in range(all_n):
        # Randomly select one of those base speeds
        selected_speed = np.random.choice(preset_speeds)
        # Load the corresponding .json preset from placo_presets
        preset_file = os.path.join(presets_dir, f"{selected_speed}.json")
        if not os.path.isfile(preset_file):
            raise FileNotFoundError(f"Preset file not found: {preset_file}")

        with open(preset_file, "r") as file:
            data = json.load(file)

        if args.sweep:
            dx_idx = i % len(dxs)
            dy_idx = (i // len(dxs)) % len(dys)
            dtheta_idx = (i // (len(dxs) * len(dys))) % len(dthetas)

            data["dx"] = round(dxs[dx_idx], 2)
            data["dy"] = round(dys[dy_idx], 2)
            data["dtheta"] = round(dthetas[dtheta_idx], 2)
        else:
            # Randomize dx, dy, dtheta within the specified max ranges
            data["dx"] = round(
                np.random.uniform(dx_max[0], dx_max[1]) * np.random.choice([-1, 1]), 2
            )
            data["dy"] = round(
                np.random.uniform(dy_max[0], dy_max[1]) * np.random.choice([-1, 1]), 2
            )
            data["dtheta"] = round(
                np.random.uniform(dtheta_max[0], dtheta_max[1])
                * np.random.choice([-1, 1]),
                2,
            )

        # Write to tmp_preset
        tmp_preset = os.path.join(tmp_dir, f"{i}_{selected_speed}.json")
        with open(tmp_preset, "w") as out_file:
            json.dump(data, out_file, indent=4)

        # Call gait_generator (no bdx-specific arguments, references removed)
        cmd = [
            "python",
            f"{SCRIPT_PATH}/../open_duck_reference_motion_generator/gait_generator.py",
            "--duck",
            args.duck,
            "--preset",
            tmp_preset,
            "--name",
            str(i),
            "--output_dir",
            args.output_dir,
            # "--meshcat_viz"
        ]
        log_file = None if args.verbose else os.path.join(log_dir, f"{i}.log")
        commands.append((cmd, log_file))

    # # ---------------------------------------------------------------
    # # 4. Generate gait motions for slow/medium/fast speeds
    # # ---------------------------------------------------------------
    # for gait_speed_key, gait_speed_val in gait_speeds.items():
    #     for gait_motion in gait_motions:
    #         # Base preset is e.g. "slow.json", "medium.json", or "fast.json"
    #         base_preset_path = os.path.join(presets_dir, f"{gait_speed_key}.json")
    #         if not os.path.isfile(base_preset_path):
    #             print(f"Skipping: no preset file {base_preset_path}")
    #             continue

    #         with open(base_preset_path, "r") as file:
    #             data = json.load(file)

    #         # Modify dx/dy/dtheta depending on the motion
    #         if gait_speed_key == "slow" and gait_motion == "standing":
    #             # Do nothing for slow standing
    #             pass
    #         elif gait_motion == "forward":
    #             data["dx"] = gait_speed_val
    #         elif gait_motion == "backward":
    #             data["dx"] = -gait_speed_val
    #         elif gait_motion == "left":
    #             data["dy"] = gait_speed_val
    #         elif gait_motion == "right":
    #             data["dy"] = -gait_speed_val
    #         elif gait_motion == "dia_forward":
    #             data["dx"] = gait_speed_val / 2
    #             data["dy"] = gait_speed_val / 2
    #         elif gait_motion == "dia_backward":
    #             data["dx"] = -gait_speed_val / 2
    #             data["dy"] = -gait_speed_val / 2
    #         elif gait_motion == "ang_left":
    #             data["dtheta"] = gait_speed_val
    #         elif gait_motion == "ang_right":
    #             data["dtheta"] = -gait_speed_val

    #         # Save to a temporary preset
    #         motion_name = f"{gait_motion}_{gait_speed_key}"
    #         motion_preset_path = os.path.join(
    #             tmp_dir, f"motion_preset_{motion_name}.json"
    #         )
    #         with open(motion_preset_path, "w") as out_file:
    #             json.dump(data, out_file, indent=4)

    #         # Run gait_generator
    #         cmd = [
    #             "python",
    #             f"{SCRIPT_PATH}/../open_duck_reference_motion_generator/gait_generator.py",
    #             "--duck",
    #             args.duck,
    #             "--preset",
    #             motion_preset_path,
    #             "--name",
    #             motion_name,
    #             "--output_dir",
    #             args.output_dir
    #         ]
    #         log_file = (
    #             None if args.verbose else os.path.join(log_dir, f"{motion_name}.log")
    #         )
    #         commands.append((cmd, log_file))

    # for command in commands:

    #     print(command)
    #     exit()
    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            executor.map(run_command_with_logging, commands)
    else:
        for cmd in commands:
            print(cmd)
            run_command_with_logging(cmd)

    # ---------------------------------------------------------------
    # 5. Check the JSON outputs in ../recordings; remove if out of range
    # ---------------------------------------------------------------
    totals = []
    if os.path.isdir(args.output_dir):
        for filename in os.listdir(args.output_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(args.output_dir, filename)
                with open(file_path, "r") as file:
                    data = json.load(file)

                placo_data = data.get("Placo", {})
                avg_x_vel = placo_data.get("avg_x_lin_vel", 0)
                avg_y_vel = placo_data.get("avg_y_lin_vel", 0)
                preset_name = placo_data.get("preset_name", "unknown")

                total_speed = np.sqrt(avg_x_vel**2 + avg_y_vel**2)

                # If the speeds do not fit the indicated preset name, remove
                if (
                    (preset_name == "slow" and total_speed > slow)
                    or (
                        preset_name == "medium"
                        and (total_speed <= slow or total_speed > fast)
                    )
                    or (preset_name == "fast" and total_speed <= medium)
                ):
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                else:
                    totals.append((total_speed, preset_name))
    else:
        print(f"No directory found at {args.output_dir}; skipping file checks.")
    totals = sorted(totals, key=numeric_prefix_sort_key)
    for speed, preset_name in totals:
        print(f"Preset: {preset_name}, Total Speed: {speed:.4f}")

    # ---------------------------------------------------------------
    # 6. Optional plotting of anim.npy and sim.npy
    # ---------------------------------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt

        anim_path = os.path.join(SCRIPT_PATH, "anim.npy")
        sim_path = os.path.join(SCRIPT_PATH, "sim.npy")

        if os.path.isfile(anim_path) and os.path.isfile(sim_path):
            anim = np.load(anim_path)
            sim = np.load(sim_path)
            print("anim shape:", anim.shape)
            print("sim shape:", sim.shape)

            plt.plot(anim[:, 0, 0, 2], label="anim z-pos")
            plt.plot(sim[:, 0, 0, 2], label="sim z-pos")
            plt.legend()
            plt.title("Comparison of anim & sim")
            plt.show()
        else:
            print("anim.npy or sim.npy not found; skipping plot.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMP walking animations")
    parser.add_argument(
        "--duck",
        choices=["go_bdx", "open_duck_mini", "open_duck_mini_v2", "sigmaban2019"],
        help="Duck type",
        required=True,
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of random motion files to generate.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        nargs="?",  # Makes the argument optional
        type=int,
        const=os.cpu_count(),  # Used when -j is provided without a number
        default=1,  # Default value when -j is not provided
        help=(
            "Number of parallel jobs. "
            "If -j is provided without a number, uses the number of CPU cores available. "
            "Default is 1."
        ),
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Sweep through the dx, dy, dtheta values."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Optionally plot anim.npy and sim.npy if they exist.",
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for the recordings",
        default=f"{SCRIPT_PATH}/../recordings",
    )
    args = parser.parse_args()
    main(args)
