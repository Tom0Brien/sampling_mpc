#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import argparse
import os


def plot_trajectory_data(data, output_dir="plots"):
    """
    Plot the collected trajectory data

    Args:
        data: Dictionary containing the collected data
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    joint_positions = data["joint_positions"]
    joint_velocities = data["joint_velocities"]

    # Support for updated data structure where commanded_ee_poses might be missing
    commanded_ee_poses = data.get("commanded_ee_poses")
    if commanded_ee_poses is None:
        print("No commanded_ee_poses found in data. Using only measured poses.")

    timestamps = data.get("timestamps")

    # Check if we have measured end-effector poses
    has_measured_poses = (
        "measured_ee_poses" in data and data["measured_ee_poses"] is not None
    )
    if has_measured_poses:
        measured_ee_poses = data["measured_ee_poses"]
        print(f"Found measured end-effector poses in data")
    else:
        print(f"No measured end-effector poses found in data")

    # Print data shapes for debugging
    print(f"Data shapes:")
    print(f"  Joint positions: {joint_positions.shape}")
    print(f"  Joint velocities: {joint_velocities.shape}")
    if commanded_ee_poses is not None:
        print(f"  Commanded end-effector poses: {commanded_ee_poses.shape}")
    if has_measured_poses:
        print(f"  Measured end-effector poses: {measured_ee_poses.shape}")
    if timestamps is not None:
        print(f"  Timestamps: {timestamps.shape}")

    # Create consistent x_values for all plots
    # For each array, create the appropriate x_values with the same length
    joint_indices = np.arange(len(joint_positions))

    if commanded_ee_poses is not None:
        cmd_pose_indices = np.arange(len(commanded_ee_poses))

    if has_measured_poses:
        measured_pose_indices = np.arange(len(measured_ee_poses))

    if timestamps is not None:
        # If timestamps available, interpolate to get consistent time values
        if len(timestamps) == len(joint_positions):
            joint_times = timestamps - timestamps[0]
        else:
            # Create artificial timestamps based on proportional positioning
            joint_times = np.linspace(0, 1, len(joint_positions))

        if commanded_ee_poses is not None and len(timestamps) == len(
            commanded_ee_poses
        ):
            cmd_pose_times = timestamps - timestamps[0]
        elif commanded_ee_poses is not None:
            # Create artificial timestamps based on proportional positioning
            cmd_pose_times = np.linspace(0, 1, len(commanded_ee_poses))

        if has_measured_poses:
            if len(timestamps) == len(measured_ee_poses):
                measured_pose_times = timestamps - timestamps[0]
            else:
                measured_pose_times = np.linspace(0, 1, len(measured_ee_poses))

        time_label = "Time (s)"
    else:
        # Use indices if no timestamps
        joint_times = joint_indices
        if commanded_ee_poses is not None:
            cmd_pose_times = cmd_pose_indices
        if has_measured_poses:
            measured_pose_times = measured_pose_indices
        time_label = "Step"

    # Create figure for cartesian position plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Cartesian Position Trajectories", fontsize=16)

    # Plot cartesian positions
    for i, axis_name in enumerate(["X", "Y", "Z"]):
        if commanded_ee_poses is not None:
            axes[i].plot(
                cmd_pose_times, commanded_ee_poses[:, i], "r--", label="Commanded"
            )

        if has_measured_poses:
            axes[i].plot(
                measured_pose_times, measured_ee_poses[:, i], "b-", label="Measured"
            )

        axes[i].set_ylabel(f"{axis_name} Position (m)")
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel(time_label)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
    plt.savefig(os.path.join(output_dir, "cartesian_position.png"))

    # Create figure for cartesian orientation plots (using Euler angles for visualization)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Cartesian Orientation Trajectories (Euler Angles)", fontsize=16)

    # Convert quaternions to Euler angles for easier visualization
    if commanded_ee_poses is not None:
        cmd_euler_angles = np.array(
            [Rotation.from_quat(q).as_euler("xyz") for q in commanded_ee_poses[:, 3:7]]
        )

    if has_measured_poses:
        measured_euler_angles = np.array(
            [Rotation.from_quat(q).as_euler("xyz") for q in measured_ee_poses[:, 3:7]]
        )

    # Plot Euler angles
    for i, angle_name in enumerate(["Roll", "Pitch", "Yaw"]):
        if commanded_ee_poses is not None:
            axes[i].plot(
                cmd_pose_times, cmd_euler_angles[:, i], "r--", label="Commanded"
            )

        if has_measured_poses:
            axes[i].plot(
                measured_pose_times, measured_euler_angles[:, i], "b-", label="Measured"
            )

        axes[i].set_ylabel(f"{angle_name} (rad)")
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel(time_label)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
    plt.savefig(os.path.join(output_dir, "cartesian_orientation.png"))

    # Create figure for joint positions (first 7 joints)
    n_joints = min(7, joint_positions.shape[1])
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 15), sharex=True)
    if n_joints == 1:
        axes = [axes]  # Make axes always iterable
    fig.suptitle("Joint Position Trajectories", fontsize=16)

    # Plot joint positions
    for i in range(n_joints):
        axes[i].plot(joint_times, joint_positions[:, i])
        axes[i].set_ylabel(f"Joint {i + 1} (rad)")
        axes[i].grid(True)

    axes[-1].set_xlabel(time_label)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
    plt.savefig(os.path.join(output_dir, "joint_positions.png"))

    # Create figure for joint velocities (first 7 joints)
    n_joints = min(7, joint_velocities.shape[1])
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 15), sharex=True)
    if n_joints == 1:
        axes = [axes]  # Make axes always iterable
    fig.suptitle("Joint Velocity Trajectories", fontsize=16)

    # Plot joint velocities
    for i in range(n_joints):
        axes[i].plot(joint_times, joint_velocities[:, i])
        axes[i].set_ylabel(f"Joint {i + 1} Vel (rad/s)")
        axes[i].grid(True)

    axes[-1].set_xlabel(time_label)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
    plt.savefig(os.path.join(output_dir, "joint_velocities.png"))

    # Create 3D plot of the trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot commanded trajectory if available
    if commanded_ee_poses is not None:
        ax.plot(
            commanded_ee_poses[:, 0],
            commanded_ee_poses[:, 1],
            commanded_ee_poses[:, 2],
            "r--",
            label="Commanded",
        )

    # Plot measured trajectory if available
    if has_measured_poses:
        ax.plot(
            measured_ee_poses[:, 0],
            measured_ee_poses[:, 1],
            measured_ee_poses[:, 2],
            "b-",
            label="Measured",
        )

    # Add scatter points for start and end positions
    if has_measured_poses:
        # Use measured poses for markers
        ax.scatter(
            measured_ee_poses[0, 0],
            measured_ee_poses[0, 1],
            measured_ee_poses[0, 2],
            c="g",
            marker="o",
            s=100,
            label="Start",
        )
        ax.scatter(
            measured_ee_poses[-1, 0],
            measured_ee_poses[-1, 1],
            measured_ee_poses[-1, 2],
            c="r",
            marker="o",
            s=100,
            label="End",
        )
    elif commanded_ee_poses is not None:
        # Fall back to commanded poses
        ax.scatter(
            commanded_ee_poses[0, 0],
            commanded_ee_poses[0, 1],
            commanded_ee_poses[0, 2],
            c="g",
            marker="o",
            s=100,
            label="Start",
        )
        ax.scatter(
            commanded_ee_poses[-1, 0],
            commanded_ee_poses[-1, 1],
            commanded_ee_poses[-1, 2],
            c="r",
            marker="o",
            s=100,
            label="End",
        )
    else:
        # No pose data available
        print("No pose data available for 3D plot")
        plt.close(fig)
        return

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Cartesian Trajectory")
    ax.legend()

    # Set equal aspect ratio for 3D plot
    # Use the poses with the widest range (either commanded or measured)
    if has_measured_poses and commanded_ee_poses is not None:
        x_range_cmd = commanded_ee_poses[:, 0].max() - commanded_ee_poses[:, 0].min()
        y_range_cmd = commanded_ee_poses[:, 1].max() - commanded_ee_poses[:, 1].min()
        z_range_cmd = commanded_ee_poses[:, 2].max() - commanded_ee_poses[:, 2].min()

        x_range_meas = measured_ee_poses[:, 0].max() - measured_ee_poses[:, 0].min()
        y_range_meas = measured_ee_poses[:, 1].max() - measured_ee_poses[:, 1].min()
        z_range_meas = measured_ee_poses[:, 2].max() - measured_ee_poses[:, 2].min()

        x_range = max(x_range_cmd, x_range_meas)
        y_range = max(y_range_cmd, y_range_meas)
        z_range = max(z_range_cmd, z_range_meas)

        max_range = max(x_range, y_range, z_range) / 2.0

        # Compute midpoints considering both sets of poses
        mid_x = (
            commanded_ee_poses[:, 0].max()
            + commanded_ee_poses[:, 0].min()
            + measured_ee_poses[:, 0].max()
            + measured_ee_poses[:, 0].min()
        ) / 4
        mid_y = (
            commanded_ee_poses[:, 1].max()
            + commanded_ee_poses[:, 1].min()
            + measured_ee_poses[:, 1].max()
            + measured_ee_poses[:, 1].min()
        ) / 4
        mid_z = (
            commanded_ee_poses[:, 2].max()
            + commanded_ee_poses[:, 2].min()
            + measured_ee_poses[:, 2].max()
            + measured_ee_poses[:, 2].min()
        ) / 4
    elif has_measured_poses:
        # Only use measured poses
        x_range = measured_ee_poses[:, 0].max() - measured_ee_poses[:, 0].min()
        y_range = measured_ee_poses[:, 1].max() - measured_ee_poses[:, 1].min()
        z_range = measured_ee_poses[:, 2].max() - measured_ee_poses[:, 2].min()

        max_range = max(x_range, y_range, z_range) / 2.0

        mid_x = (measured_ee_poses[:, 0].max() + measured_ee_poses[:, 0].min()) / 2
        mid_y = (measured_ee_poses[:, 1].max() + measured_ee_poses[:, 1].min()) / 2
        mid_z = (measured_ee_poses[:, 2].max() + measured_ee_poses[:, 2].min()) / 2
    elif commanded_ee_poses is not None:
        # Only use commanded poses
        x_range = commanded_ee_poses[:, 0].max() - commanded_ee_poses[:, 0].min()
        y_range = commanded_ee_poses[:, 1].max() - commanded_ee_poses[:, 1].min()
        z_range = commanded_ee_poses[:, 2].max() - commanded_ee_poses[:, 2].min()

        max_range = max(x_range, y_range, z_range) / 2.0

        mid_x = (commanded_ee_poses[:, 0].max() + commanded_ee_poses[:, 0].min()) / 2
        mid_y = (commanded_ee_poses[:, 1].max() + commanded_ee_poses[:, 1].min()) / 2
        mid_z = (commanded_ee_poses[:, 2].max() + commanded_ee_poses[:, 2].min()) / 2
    else:
        # No pose data available
        print("No pose data available for 3D plot")
        plt.close(fig)
        return

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3d_trajectory.png"))

    # If we have both commanded and measured poses, create tracking error plots only
    if has_measured_poses and commanded_ee_poses is not None:
        try:
            # Interpolate commanded trajectory to match measured time points
            from scipy.interpolate import interp1d

            # Create interpolation functions for commanded trajectory
            interp_cmd_pos = [
                interp1d(cmd_pose_times, commanded_ee_poses[:, i]) for i in range(3)
            ]

            # For orientation, need to interpolate Euler angles
            cmd_euler_angles = np.array(
                [
                    Rotation.from_quat(q).as_euler("xyz")
                    for q in commanded_ee_poses[:, 3:7]
                ]
            )
            measured_euler_angles = np.array(
                [
                    Rotation.from_quat(q).as_euler("xyz")
                    for q in measured_ee_poses[:, 3:7]
                ]
            )
            interp_cmd_euler = [
                interp1d(cmd_pose_times, cmd_euler_angles[:, i]) for i in range(3)
            ]

            # Create tracking error plots
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            fig.suptitle("Position Tracking Error", fontsize=16)

            # For each axis, calculate and plot tracking error
            for i, axis_name in enumerate(["X", "Y", "Z"]):
                # Interpolate commanded position to measured time points
                cmd_interp = interp_cmd_pos[i](measured_pose_times)
                # Calculate error
                error = measured_ee_poses[:, i] - cmd_interp
                # Plot error
                axes[i].plot(measured_pose_times, error)
                axes[i].set_ylabel(f"{axis_name} Error (m)")
                axes[i].grid(True)

            axes[-1].set_xlabel(time_label)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, "position_tracking_error.png"))

            # Create orientation error plot
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            fig.suptitle("Orientation Tracking Error", fontsize=16)

            for i, angle_name in enumerate(["Roll", "Pitch", "Yaw"]):
                # Interpolate commanded orientation to measured time points
                cmd_interp = interp_cmd_euler[i](measured_pose_times)
                # Calculate error
                error = measured_euler_angles[:, i] - cmd_interp
                # Plot error
                axes[i].plot(measured_pose_times, error)
                axes[i].set_ylabel(f"{angle_name} Error (rad)")
                axes[i].grid(True)

            axes[-1].set_xlabel(time_label)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, "orientation_tracking_error.png"))

        except Exception as e:
            print(f"Error creating tracking error plots: {e}")
    elif has_measured_poses and commanded_ee_poses is None:
        print("Only measured poses available - skipping tracking error plots")

    print(f"Plots saved to directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot collected Franka robot data")
    parser.add_argument("data_file", type=str, help="Path to the .npy data file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sys_id/plots",
        help="Directory to save plot images",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_file}")
    data = np.load(args.data_file, allow_pickle=True).item()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot data
    plot_trajectory_data(data, args.output_dir)

    print("Plotting complete!")


if __name__ == "__main__":
    main()
