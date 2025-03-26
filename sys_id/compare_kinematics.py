#!/usr/bin/env python3
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mujoco

from hydrax.tasks.franka_reach import FrankaReach
from hydrax.task_base import ControlMode


def load_data(data_file):
    """Load collected data from file."""
    print(f"Loading data from {data_file}")
    data = np.load(data_file, allow_pickle=True).item()

    # Check if we have measured poses
    if "measured_ee_poses" not in data:
        print("Error: No measured end-effector poses found in the file")
        return None

    return data


def compute_mujoco_poses(data):
    """Compute end-effector poses using MuJoCo forward kinematics."""
    # Create a FrankaReach task to get the model
    task = FrankaReach(control_mode=ControlMode.CARTESIAN_SIMPLE_VI)
    model = task.mj_model
    sim_data = mujoco.MjData(model)

    # Get the end-effector site ID
    ee_site_id = model.site("gripper").id

    # Get joint positions
    joint_positions = data["joint_positions"]

    # Compute poses for each timestep
    n_steps = len(joint_positions)
    computed_poses = []

    print(f"Computing poses for {n_steps} timesteps...")

    for i in range(n_steps):
        # Set the robot state in MuJoCo
        sim_data.qpos[:7] = joint_positions[i, :7]

        # Run forward kinematics
        mujoco.mj_kinematics(model, sim_data)

        # Get end-effector position and orientation
        pos = sim_data.site_xpos[ee_site_id].copy()
        rot_matrix = sim_data.site_xmat[ee_site_id].reshape(3, 3)
        quat = Rotation.from_matrix(rot_matrix).as_quat()

        # Combine position and orientation
        pose = np.concatenate([pos, quat])
        computed_poses.append(pose)

        if i % 100 == 0:
            print(f"  {i}/{n_steps} timesteps processed")

    return np.array(computed_poses)


def plot_pose_comparison(
    measured_poses, computed_poses, output_dir="sys_id/pose_comparison"
):
    """Create comparative plots between measured and computed poses."""
    os.makedirs(output_dir, exist_ok=True)

    # Make sure sizes match
    min_len = min(len(measured_poses), len(computed_poses))
    measured_poses = measured_poses[:min_len]
    computed_poses = computed_poses[:min_len]

    # Create time array
    time = np.arange(min_len) / 100.0  # Assuming 100Hz sampling rate

    # Plot position comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].set_title("End-Effector Position Comparison")

    for i, axis in enumerate(["X", "Y", "Z"]):
        axes[i].plot(time, measured_poses[:, i], "b-", label="Measured")
        axes[i].plot(time, computed_poses[:, i], "r--", label="MuJoCo")
        axes[i].set_ylabel(f"{axis} Position (m)")
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel("Time (s)")
    plt.savefig(os.path.join(output_dir, "position_comparison.png"))

    # Plot orientation comparison (Euler angles)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].set_title("End-Effector Orientation Comparison (Euler Angles)")

    # Convert quaternions to Euler angles
    measured_euler = np.array(
        [
            Rotation.from_quat(q).as_euler("xyz", degrees=True)
            for q in measured_poses[:, 3:7]
        ]
    )
    computed_euler = np.array(
        [
            Rotation.from_quat(q).as_euler("xyz", degrees=True)
            for q in computed_poses[:, 3:7]
        ]
    )

    for i, angle in enumerate(["Roll", "Pitch", "Yaw"]):
        axes[i].plot(time, measured_euler[:, i], "b-", label="Measured")
        axes[i].plot(time, computed_euler[:, i], "r--", label="MuJoCo")
        axes[i].set_ylabel(f"{angle} (degrees)")
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel("Time (s)")
    plt.savefig(os.path.join(output_dir, "orientation_comparison.png"))

    # Calculate and print position error statistics per coordinate
    print("\nPosition Error Statistics:")
    print("-" * 50)
    print("Coordinate | Mean Error (m) | Max Error (m) | RMS Error (m)")
    print("-" * 50)

    for i, axis in enumerate(["X", "Y", "Z"]):
        coord_error = np.abs(measured_poses[:, i] - computed_poses[:, i])
        mean_error = np.mean(coord_error)
        max_error = np.max(coord_error)
        rms_error = np.sqrt(np.mean(coord_error**2))
        print(
            f"{axis:^10} | {mean_error:^12.4f} | {max_error:^12.4f} | {rms_error:^12.4f}"
        )

    # Overall position error
    pos_error = np.linalg.norm(measured_poses[:, :3] - computed_poses[:, :3], axis=1)
    print("-" * 50)
    print(
        f"Overall   | {np.mean(pos_error):^12.4f} | {np.max(pos_error):^12.4f} | {np.sqrt(np.mean(pos_error**2)):^12.4f}"
    )

    # Calculate and print orientation error statistics
    print("\nOrientation Error Statistics (degrees):")
    print("-" * 50)
    print("Angle | Mean Error | Max Error | RMS Error")
    print("-" * 50)

    for i, angle in enumerate(["Roll", "Pitch", "Yaw"]):
        angle_error = np.abs(measured_euler[:, i] - computed_euler[:, i])
        mean_error = np.mean(angle_error)
        max_error = np.max(angle_error)
        rms_error = np.sqrt(np.mean(angle_error**2))
        print(
            f"{angle:^6} | {mean_error:^10.2f} | {max_error:^9.2f} | {rms_error:^9.2f}"
        )

    # Overall orientation error (quaternion difference)
    print("\nPlots saved to", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Compare measured end-effector poses with MuJoCo forward kinematics"
    )
    parser.add_argument("data_file", type=str, help="Path to the .npy data file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sys_id/plots",
        help="Directory to save comparison plots",
    )
    args = parser.parse_args()

    # Load the collected data
    data = load_data(args.data_file)
    if data is None:
        return

    # Get measured poses
    measured_poses = data["measured_ee_poses"]

    # Compute poses using MuJoCo forward kinematics
    computed_poses = compute_mujoco_poses(data)

    # Create comparative plots and print statistics
    plot_pose_comparison(measured_poses, computed_poses, args.output_dir)


if __name__ == "__main__":
    main()
