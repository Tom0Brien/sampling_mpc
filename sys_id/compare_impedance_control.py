#!/usr/bin/env python3
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mujoco

from hydrax.tasks.franka_reach import FrankaReach
from hydrax.task_base import ControlMode
from hydrax import ROOT
from hydrax.controllers.impedance_controllers import impedance_control


def load_data(data_file):
    """Load collected data from file.

    Args:
        data_file: Path to the .npy data file

    Returns:
        Dictionary with collected data
    """
    print(f"Loading data from {data_file}")
    data = np.load(data_file, allow_pickle=True).item()

    # Print info about the data
    print("Data contains:")
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(f"  {key}: shape {data[key].shape}")
        else:
            print(f"  {key}")

    # Check if we have joint effort data
    if "joint_efforts" not in data or data["joint_efforts"] is None:
        print("Warning: No joint effort (torque) data found in the file")

    return data


def compute_mujoco_torques(data):
    """Compute torques using MuJoCo's impedance controller for comparison.

    Args:
        data: Dictionary with collected data

    Returns:
        Computed torques from MuJoCo controller
    """
    # Create a FrankaReach task to get the model
    task = FrankaReach(control_mode=ControlMode.CARTESIAN_SIMPLE_VI)
    model = task.mj_model
    sim_data = mujoco.MjData(model)

    # Get the end-effector site ID
    ee_site_id = model.site("gripper").id

    # Extract controller parameters
    controller_params = data["controller_params"]
    trans_stiffness = controller_params["translational_stiffness"]
    rot_stiffness = controller_params["rotational_stiffness"]
    nullspace_stiffness = controller_params.get("nullspace_stiffness", 0.0)
    print("nullspace_stiffness", nullspace_stiffness)

    # Create gain matrices (diagonal matrices with stiffness values)
    Kp = np.zeros(6)
    Kp[:3] = trans_stiffness  # Translational stiffness for x, y, z
    Kp[3:] = rot_stiffness  # Rotational stiffness for roll, pitch, yaw

    # Critical damping: Kd = 2 * sqrt(Kp)
    Kd = 2.0 * np.sqrt(Kp)

    # Use diagonal matrices for Kp and Kd
    Kp_diag = np.diag(Kp)
    Kd_diag = np.diag(Kd)

    # Default desired nullspace configuration (home position)
    q_d_nullspace = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0, 0])

    # Get commanded poses and joint states
    joint_positions = data["joint_positions"]
    joint_velocities = data["joint_velocities"]
    commanded_ee_poses = data["commanded_ee_poses"]

    # Use measured poses if available, otherwise use commanded
    if "measured_ee_poses" in data and data["measured_ee_poses"] is not None:
        measured_ee_poses = data["measured_ee_poses"]
    else:
        measured_ee_poses = None

    # Compute torques for each timestep
    n_steps = min(len(joint_positions), len(commanded_ee_poses))
    computed_torques = []

    print(f"Computing torques for {n_steps} timesteps...")

    for i in range(n_steps):
        # Set the robot state in MuJoCo
        sim_data.qpos[:7] = joint_positions[i, :7]
        sim_data.qvel[:7] = joint_velocities[i, :7]

        # Run forward kinematics to update positions
        mujoco.mj_kinematics(model, sim_data)

        # Get the commanded pose
        cmd_pose = commanded_ee_poses[i]

        # Calculate impedance control torques
        tau = impedance_control(
            model,
            sim_data,
            cmd_pose[:3],  # desired position
            Rotation.from_quat(cmd_pose[3:7]).as_euler("xyz"),  # desired orientation
            Kp_diag,  # stiffness matrix
            Kd_diag,  # damping matrix
            nullspace_stiffness,
            q_d_nullspace,
            ee_site_id,
        )

        # Store computed torques
        computed_torques.append(tau[:7])  # Just the 7 robot joints

        # Print progress
        if i % 100 == 0:
            print(f"  {i}/{n_steps} timesteps processed")

    return np.array(computed_torques)


def plot_torque_comparison(
    real_torques, computed_torques, output_dir="sys_id/torque_comparison"
):
    """Create comparative plots between real and computed torques.

    Args:
        real_torques: Measured torques from the robot
        computed_torques: Torques computed from MuJoCo controller
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Make sure sizes match
    min_len = min(len(real_torques), len(computed_torques))
    real_torques = real_torques[:min_len]
    computed_torques = computed_torques[:min_len]

    # Create time array for plotting
    time = np.arange(min_len) / 100.0  # Assuming 100Hz sampling rate, adjust if needed

    # Plot torques for each joint
    n_joints = min(7, real_torques.shape[1])

    # Create figure for all joint torques
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 16), sharex=True)
    if n_joints == 1:
        axes = [axes]  # Make axes always iterable
    fig.suptitle("Joint Torque Comparison: Real vs MuJoCo", fontsize=16)

    for i in range(n_joints):
        axes[i].plot(time, real_torques[:, i], "b-", label="Measured")
        axes[i].plot(time, computed_torques[:, i], "r--", label="MuJoCo Computed")
        axes[i].set_ylabel(f"Joint {i + 1} Torque (Nm)")
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for title
    plt.savefig(os.path.join(output_dir, "joint_torque_comparison.png"))

    # Calculate absolute torque error
    torque_error = np.abs(real_torques - computed_torques)

    # Plot error for each joint
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 16), sharex=True)
    if n_joints == 1:
        axes = [axes]  # Make axes always iterable
    fig.suptitle("Absolute Torque Error: |Measured - Computed|", fontsize=16)

    for i in range(n_joints):
        axes[i].plot(time, torque_error[:, i])
        axes[i].set_ylabel(f"Joint {i + 1} Error (Nm)")
        axes[i].grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for title
    plt.savefig(os.path.join(output_dir, "joint_torque_error.png"))

    # Calculate and plot overall statistics
    mean_error = np.mean(torque_error, axis=0)
    std_error = np.std(torque_error, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    joint_labels = [f"Joint {i + 1}" for i in range(n_joints)]
    x = np.arange(len(joint_labels))
    width = 0.35

    ax.bar(x, mean_error, width, yerr=std_error, label="Mean Absolute Error", capsize=5)
    ax.set_ylabel("Torque Error (Nm)")
    ax.set_title("Mean Absolute Torque Error by Joint")
    ax.set_xticks(x)
    ax.set_xticklabels(joint_labels)
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "torque_error_statistics.png"))

    # Print statistics
    print("\nTorque Error Statistics:")
    print("Joint | Mean Abs Error (Nm) | Std Deviation")
    print("-" * 45)
    for i in range(n_joints):
        print(f"  {i + 1}   |     {mean_error[i]:8.4f}      |    {std_error[i]:8.4f}")

    # Also compute RMSE
    rmse = np.sqrt(np.mean(np.square(real_torques - computed_torques), axis=0))
    print("\nRMSE by Joint:")
    for i in range(n_joints):
        print(f"  Joint {i + 1}: {rmse[i]:.4f} Nm")

    print(f"\nPlots saved to {output_dir}")
    return {"mean_error": mean_error, "std_error": std_error, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser(
        description="Compare collected torques with MuJoCo impedance controller"
    )
    parser.add_argument("data_file", type=str, help="Path to the .npy data file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sys_id/torque_comparison",
        help="Directory to save comparison plots",
    )
    args = parser.parse_args()

    # Load the collected data
    data = load_data(args.data_file)

    # Check if the data has joint effort (torque) data
    if "joint_efforts" not in data or data["joint_efforts"] is None:
        print("Error: No joint effort (torque) data in the provided file")
        return

    # Extract the real torques
    real_torques = data["joint_efforts"]

    # Compute torques using MuJoCo impedance controller
    computed_torques = compute_mujoco_torques(data)

    # Create comparative plots and print statistics
    plot_torque_comparison(real_torques, computed_torques, args.output_dir)


if __name__ == "__main__":
    main()
