#!/usr/bin/env python3
import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import mujoco
import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.franka_reach import FrankaReach
from hydrax.task_base import GainOptimizationMode


class FrankaMujocoSimulator:
    """Simulator for the Franka robot in MuJoCo for system identification."""

    def __init__(self, model_path=None):
        """Initialize the simulator with a MuJoCo model.

        Args:
            model_path: Path to the MuJoCo XML model file. If None, uses the default
                        FrankaReach model.
        """
        # Create a FrankaReach task to get the model
        self.task = FrankaReach(gain_mode=GainOptimizationMode.SIMPLE)
        self.mj_model = self.task.mj_model

        # Keep a copy of the original model parameters that we will modify
        self.default_damp = np.copy(self.mj_model.dof_damping)
        self.default_gain_scaling = 1.0

        # Print some model information
        print(
            f"Model has {self.mj_model.nq} position DoFs and {self.mj_model.nv} velocity DoFs"
        )
        print(f"Joint damping values: {self.default_damp}")

    def update_parameters(self, params):
        """Update the model parameters based on the optimization parameters.

        Args:
            params: List or array of parameters:
                   - First n parameters are joint damping multipliers
                   - Last parameter is controller gain scaling

        Returns:
            Updated MuJoCo model and initial data
        """
        # Extract parameters
        n_damp = len(self.default_damp)
        damp_multipliers = params[:n_damp]
        gain_scaling = params[n_damp]

        # Create a copy of the model
        model = self.mj_model

        # Update joint damping (multiply original values by scaling factors)
        model.dof_damping = self.default_damp * damp_multipliers

        # Return the updated model and controller gain scaling
        return model, gain_scaling

    def simulate_trajectory(
        self,
        model,
        gain_scaling,
        commanded_ee_poses,
        controller_params,
        duration,
        initial_state=None,
    ):
        """Simulate a trajectory with the given parameters.

        Args:
            model: MuJoCo model with updated parameters
            gain_scaling: Controller gain scaling factor
            commanded_ee_poses: Array of commanded Cartesian poses [x, y, z, qx, qy, qz, qw]
            controller_params: Controller parameters dict with keys:
                               - translational_stiffness
                               - rotational_stiffness
                               - nullspace_stiffness
            duration: Duration of the simulation in seconds
            initial_state: Initial joint positions and velocities, if None uses default

        Returns:
            Dictionary with simulated trajectory data
        """
        # Create a data instance for simulation
        data = mujoco.MjData(model)

        # Set initial state if provided
        if initial_state is not None:
            data.qpos = initial_state["qpos"]
            data.qvel = initial_state["qvel"]
        else:
            # Default initial configuration for Franka
            data.qpos[:7] = [-0.196, -0.189, 0.182, -2.1, 0.0378, 1.91, 0.756, 0, 0]
            data.qvel[:9] = 0.0

        # Run mujoco forward
        mujoco.mj_kinematics(model, data)

        # Extract controller parameters and apply scaling
        trans_stiffness = controller_params["translational_stiffness"] * gain_scaling
        rot_stiffness = controller_params["rotational_stiffness"] * gain_scaling

        # Calculate corresponding damping from stiffness (critical damping assumption)
        trans_damping = 2.0 * np.sqrt(trans_stiffness)  # Critical damping
        rot_damping = 2.0 * np.sqrt(rot_stiffness)  # Critical damping

        # Update controller gains in the model
        # For each cartesian DoF, set the appropriate gain
        for i in range(3):  # x, y, z translation
            model.actuator_gainprm[i, 1] = trans_stiffness  # Kp
            model.actuator_gainprm[i, 2] = trans_damping  # Kd

        for i in range(3, 6):  # roll, pitch, yaw rotation
            model.actuator_gainprm[i, 1] = rot_stiffness  # Kp
            model.actuator_gainprm[i, 2] = rot_damping  # Kd

        # Determine the time step and number of steps
        timestep = model.opt.timestep
        n_steps = int(duration / timestep)

        # Interpolate commanded poses to match simulation timesteps
        sim_times = np.linspace(0, duration, n_steps)
        cmd_times = np.linspace(0, duration, len(commanded_ee_poses))

        # Create storage for trajectory data
        joint_positions = []
        joint_velocities = []
        simulated_ee_poses = []

        # Run the simulation
        for i in range(n_steps):
            # Get the current simulation time
            current_time = i * timestep

            # Find the closest commanded pose or interpolate
            idx = np.argmin(np.abs(cmd_times - current_time))
            cmd_pose = commanded_ee_poses[idx]

            # Transform the desired pose into the mujoco reference frame
            rRBb = data.site_xpos[self.task.reference_id]
            Rbr = data.site_xmat[self.task.reference_id].reshape(3, 3)

            # Transform the desired pose into the mujoco reference frame
            rGBb = cmd_pose[:3]
            rGRr = Rbr.T @ (rGBb - rRBb)
            Rbg = Rotation.from_quat(cmd_pose[3:7]).as_matrix()
            Rrg = Rbr.T @ Rbg
            rpy = Rotation.from_matrix(Rrg).as_euler("xyz")

            # Set the target pose in the controller
            data.ctrl[:3] = rGRr
            data.ctrl[3:6] = rpy

            # Step the simulation
            mujoco.mj_step(model, data)

            # Every few steps, collect data
            if i % 10 == 0:  # Adjust sampling rate as needed
                # Calculate end-effector pose (forward kinematics)
                mujoco.mj_kinematics(model, data)

                # Get end-effector site position and orientation
                ee_site_id = model.site("gripper").id
                ee_pos = data.site_xpos[ee_site_id].copy()
                ee_ori_mat = data.site_xmat[ee_site_id].reshape(3, 3)
                ee_quat = Rotation.from_matrix(ee_ori_mat).as_quat()

                # Store data
                joint_positions.append(data.qpos[:7].copy())
                joint_velocities.append(data.qvel[:7].copy())
                simulated_ee_poses.append(np.concatenate([ee_pos, ee_quat]))

        # Convert lists to numpy arrays
        result = {
            "joint_positions": np.array(joint_positions),
            "joint_velocities": np.array(joint_velocities),
            "simulated_ee_poses": np.array(simulated_ee_poses),
            "commanded_ee_poses": commanded_ee_poses,
        }

        return result


class SystemIdentifier:
    """System identification for the Franka robot using collected data."""

    def __init__(self, data_file, simulator=None):
        """Initialize the system identifier.

        Args:
            data_file: Path to the collected data file (.npy)
            simulator: FrankaMujocoSimulator instance, or None to create a new one
        """
        # Load the collected data
        print(f"Loading data from {data_file}")
        self.real_data = np.load(data_file, allow_pickle=True).item()

        # Create or use the provided simulator
        self.simulator = simulator if simulator else FrankaMujocoSimulator()

        # Extract relevant information from the data
        self.commanded_ee_poses = self.real_data["commanded_ee_poses"]
        self.measured_ee_poses = self.real_data[
            "measured_ee_poses"
        ]  # Using existing naming from data file
        self.controller_params = self.real_data["controller_params"]

        # Calculate trajectory duration
        if "timestamps" in self.real_data and self.real_data["timestamps"] is not None:
            timestamps = self.real_data["timestamps"]
            self.duration = timestamps[-1] - timestamps[0]
        else:
            # Assume 10Hz control rate if no timestamps
            self.duration = len(self.commanded_ee_poses) / 10.0

        print(f"Trajectory duration: {self.duration:.2f} seconds")
        print(f"Controller parameters: {self.controller_params}")
        print(f"Number of commanded poses: {len(self.commanded_ee_poses)}")
        print(f"Number of measured poses: {len(self.measured_ee_poses)}")

        # Initial state (first joint position and velocity)
        self.initial_state = {
            "qpos": self.real_data["joint_positions"][0],
            "qvel": self.real_data["joint_velocities"][0],
        }

    def objective_function(self, params):
        """Objective function for optimization.

        Args:
            params: List or array of parameters:
                   - First n parameters are joint damping multipliers
                   - Last parameter is controller gain scaling

        Returns:
            Error metric (RMSE) between real and simulated trajectories
        """
        # Print the current parameters
        n_damp = len(self.simulator.default_damp)
        damp_multipliers = params[:n_damp]
        gain_scaling = params[n_damp]

        print(
            f"Evaluating - Damping multipliers: {damp_multipliers.round(2)}, Gain scaling: {gain_scaling:.2f}"
        )

        # Update the model parameters
        model, gain_scaling = self.simulator.update_parameters(params)

        # Simulate the trajectory
        start_time = time.time()
        sim_data = self.simulator.simulate_trajectory(
            model,
            gain_scaling,
            self.commanded_ee_poses,
            self.controller_params,
            self.duration,
            self.initial_state,
        )
        sim_time = time.time() - start_time

        # Compute error between real and simulated trajectories
        # For cartesian error, we need to resample either the real or simulated data
        # to have the same number of points
        real_poses = self.measured_ee_poses
        sim_poses = sim_data["simulated_ee_poses"]

        # If different lengths, resample the shorter trajectory
        if len(real_poses) != len(sim_poses):
            if len(real_poses) < len(sim_poses):
                # Resample simulated data to match real data length
                indices = np.linspace(0, len(sim_poses) - 1, len(real_poses)).astype(
                    int
                )
                sim_poses = sim_poses[indices]
            else:
                # Resample real data to match simulated data length
                indices = np.linspace(0, len(real_poses) - 1, len(sim_poses)).astype(
                    int
                )
                real_poses = real_poses[indices]

        # Calculate position error
        pos_error = np.linalg.norm(real_poses[:, :3] - sim_poses[:, :3], axis=1)

        # Calculate orientation error using quaternion distance
        ori_error = np.zeros(len(real_poses))
        for i in range(len(real_poses)):
            real_q = real_poses[i, 3:7]
            sim_q = sim_poses[i, 3:7]
            # Quaternion dot product gives cosine of half the rotation angle
            # Make sure quaternions have the same sign
            dot_product = np.abs(np.sum(real_q * sim_q))
            dot_product = min(dot_product, 1.0)  # Numerical stability
            # Convert to angle
            ori_error[i] = 2 * np.arccos(dot_product)

        # Combined error metric (position in meters, orientation in radians)
        # Scale orientation error to be roughly comparable to position error
        pos_weight = 1.0  # meters
        ori_weight = 0.5  # radians to meters conversion

        # Mean squared error
        mean_pos_error = np.mean(pos_error**2)
        mean_ori_error = np.mean(ori_error**2)

        # Combined error
        combined_error = np.sqrt(mean_pos_error + ori_weight * mean_ori_error)

        print(
            f"Simulation time: {sim_time:.2f}s, Error: {combined_error:.4f} "
            + f"(Pos: {np.sqrt(mean_pos_error):.4f}m, Ori: {np.sqrt(mean_ori_error):.4f}rad)"
        )

        return combined_error

    def optimize(self, method="BFGS", options=None):
        """Run the optimization process.

        Args:
            method: Optimization method for scipy.optimize.minimize
            options: Dictionary of options for the optimizer

        Returns:
            Optimization results
        """
        # Default options
        if options is None:
            options = {"maxiter": 30, "disp": True}

        # Initial parameters
        n_damp = len(self.simulator.default_damp)
        initial_params = np.ones(n_damp + 1)  # Damping multipliers + gain scaling

        # Bounds: damping multipliers between 0.1 and 10, gain scaling between 0.1 and 2.0
        bounds = [(0.1, 10.0)] * n_damp + [(0.1, 2.0)]

        # Run the optimization
        print("Starting optimization...")
        result = minimize(
            self.objective_function,
            initial_params,
            method=method,
            bounds=bounds,
            options=options,
        )

        return result

    def evaluate_and_visualize(self, params):
        """Evaluate the model with given parameters and visualize the results.

        Args:
            params: Optimized parameters

        Returns:
            Dictionary with simulation results and comparison metrics
        """
        # Update model with optimized parameters
        model, gain_scaling = self.simulator.update_parameters(params)

        # Simulate the trajectory
        sim_data = self.simulator.simulate_trajectory(
            model,
            gain_scaling,
            self.commanded_ee_poses,
            self.controller_params,
            self.duration,
            self.initial_state,
        )

        # Create comparison plots
        self._create_comparison_plots(sim_data)

        # Extract and print the optimized parameters
        n_damp = len(self.simulator.default_damp)
        damp_multipliers = params[:n_damp]
        gain_scaling = params[n_damp]

        optimized_damping = self.simulator.default_damp * damp_multipliers

        print("\nOptimized Parameters:")
        print(f"Joint damping values: {optimized_damping}")
        print(f"Controller gain scaling: {gain_scaling}")

        # Calculate the error metrics
        real_poses = self.measured_ee_poses
        sim_poses = sim_data["simulated_ee_poses"]

        # Resample if needed
        if len(real_poses) != len(sim_poses):
            if len(real_poses) < len(sim_poses):
                indices = np.linspace(0, len(sim_poses) - 1, len(real_poses)).astype(
                    int
                )
                sim_poses = sim_poses[indices]
            else:
                indices = np.linspace(0, len(real_poses) - 1, len(sim_poses)).astype(
                    int
                )
                real_poses = real_poses[indices]

        # Position error
        pos_error = np.linalg.norm(real_poses[:, :3] - sim_poses[:, :3], axis=1)

        # Orientation error
        ori_error = np.zeros(len(real_poses))
        for i in range(len(real_poses)):
            real_q = real_poses[i, 3:7]
            sim_q = sim_poses[i, 3:7]
            dot_product = np.abs(np.sum(real_q * sim_q))
            dot_product = min(dot_product, 1.0)
            ori_error[i] = 2 * np.arccos(dot_product)

        # Error metrics
        rmse_pos = np.sqrt(np.mean(pos_error**2))
        rmse_ori = np.sqrt(np.mean(ori_error**2))

        print("\nError Metrics:")
        print(f"Position RMSE: {rmse_pos:.4f} meters")
        print(f"Orientation RMSE: {rmse_ori:.4f} radians")

        # Return results
        return {
            "sim_data": sim_data,
            "optimized_damping": optimized_damping,
            "gain_scaling": gain_scaling,
            "rmse_pos": rmse_pos,
            "rmse_ori": rmse_ori,
        }

    def _create_comparison_plots(self, sim_data):
        """Create plots comparing real and simulated data.

        Args:
            sim_data: Simulated trajectory data
        """
        # Create output directory for plots
        output_dir = "sys_id/results"
        os.makedirs(output_dir, exist_ok=True)

        # Extract data
        real_poses = self.measured_ee_poses
        sim_poses = sim_data["simulated_ee_poses"]
        cmd_poses = self.commanded_ee_poses

        # If different lengths, create time vectors for interpolation
        if len(real_poses) != len(sim_poses) or len(real_poses) != len(cmd_poses):
            real_times = np.linspace(0, self.duration, len(real_poses))
            sim_times = np.linspace(0, self.duration, len(sim_poses))
            cmd_times = np.linspace(0, self.duration, len(cmd_poses))
        else:
            real_times = sim_times = cmd_times = np.linspace(
                0, self.duration, len(real_poses)
            )

        # Plot position components
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        for i, (axis_name, axis_idx) in enumerate(zip(["X", "Y", "Z"], [0, 1, 2])):
            axes[i].plot(cmd_times, cmd_poses[:, axis_idx], "g-.", label="Commanded")
            axes[i].plot(real_times, real_poses[:, axis_idx], "b-", label="Real Robot")
            axes[i].plot(
                sim_times, sim_poses[:, axis_idx], "r--", label="MuJoCo Simulation"
            )
            axes[i].set_ylabel(f"{axis_name} Position (m)")
            axes[i].grid(True)
            axes[i].legend()

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Position Comparison", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "position_comparison.png"))

        # Convert quaternions to Euler angles for visualization
        real_euler = np.array(
            [Rotation.from_quat(q).as_euler("xyz") for q in real_poses[:, 3:7]]
        )
        sim_euler = np.array(
            [Rotation.from_quat(q).as_euler("xyz") for q in sim_poses[:, 3:7]]
        )
        cmd_euler = np.array(
            [Rotation.from_quat(q).as_euler("xyz") for q in cmd_poses[:, 3:7]]
        )

        # Plot orientation components
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        for i, (axis_name, axis_idx) in enumerate(
            zip(["Roll", "Pitch", "Yaw"], [0, 1, 2])
        ):
            axes[i].plot(cmd_times, cmd_euler[:, axis_idx], "g-.", label="Commanded")
            axes[i].plot(real_times, real_euler[:, axis_idx], "b-", label="Real Robot")
            axes[i].plot(
                sim_times, sim_euler[:, axis_idx], "r--", label="MuJoCo Simulation"
            )
            axes[i].set_ylabel(f"{axis_name} (rad)")
            axes[i].grid(True)
            axes[i].legend()

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Orientation Comparison", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "orientation_comparison.png"))

        # Plot position error over time (vs. commanded and simulated)
        if len(real_poses) != len(sim_poses) or len(real_poses) != len(cmd_poses):
            # Interpolate sim_poses and cmd_poses to match real_poses times
            from scipy.interpolate import interp1d

            # Create interpolation functions
            sim_pos_interp = [interp1d(sim_times, sim_poses[:, i]) for i in range(3)]
            sim_ori_interp = [interp1d(sim_times, sim_euler[:, i]) for i in range(3)]
            cmd_pos_interp = [interp1d(cmd_times, cmd_poses[:, i]) for i in range(3)]
            cmd_ori_interp = [interp1d(cmd_times, cmd_euler[:, i]) for i in range(3)]

            # Calculate errors (relative to real robot measurements)
            sim_pos_error = np.zeros((len(real_times), 3))
            sim_ori_error = np.zeros((len(real_times), 3))
            cmd_pos_error = np.zeros((len(real_times), 3))
            cmd_ori_error = np.zeros((len(real_times), 3))

            for i in range(3):
                # Real - Simulated
                sim_pos_error[:, i] = real_poses[:, i] - sim_pos_interp[i](real_times)
                sim_ori_error[:, i] = real_euler[:, i] - sim_ori_interp[i](real_times)

                # Real - Commanded
                cmd_pos_error[:, i] = real_poses[:, i] - cmd_pos_interp[i](real_times)
                cmd_ori_error[:, i] = real_euler[:, i] - cmd_ori_interp[i](real_times)

            error_times = real_times
        else:
            sim_pos_error = real_poses[:, :3] - sim_poses[:, :3]
            sim_ori_error = real_euler - sim_euler
            cmd_pos_error = real_poses[:, :3] - cmd_poses[:, :3]
            cmd_ori_error = real_euler - cmd_euler
            error_times = real_times

        # Plot position error
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        for i, axis_name in enumerate(["X", "Y", "Z"]):
            axes[i].plot(error_times, cmd_pos_error[:, i], "g-", label="vs. Commanded")
            axes[i].plot(
                error_times, sim_pos_error[:, i], "r--", label="vs. Simulation"
            )
            axes[i].set_ylabel(f"{axis_name} Error (m)")
            axes[i].grid(True)
            axes[i].legend()

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Position Error (Real - Reference)", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "position_error.png"))

        # Plot orientation error
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        for i, axis_name in enumerate(["Roll", "Pitch", "Yaw"]):
            axes[i].plot(error_times, cmd_ori_error[:, i], "g-", label="vs. Commanded")
            axes[i].plot(
                error_times, sim_ori_error[:, i], "r--", label="vs. Simulation"
            )
            axes[i].set_ylabel(f"{axis_name} Error (rad)")
            axes[i].grid(True)
            axes[i].legend()

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Orientation Error (Real - Reference)", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "orientation_error.png"))


def main():
    parser = argparse.ArgumentParser(
        description="System identification for Franka robot"
    )
    parser.add_argument(
        "data_file", type=str, help="Path to the collected data file (.npy)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="Nelder-Mead",
        choices=["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B"],
        help="Optimization method",
    )
    parser.add_argument(
        "--max-iter", type=int, default=20, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sys_id/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate with provided parameters instead of optimizing",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Comma-separated list of parameters for evaluation",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create system identifier
    identifier = SystemIdentifier(args.data_file)

    if args.evaluate_only and args.params:
        # Parse parameters
        params = np.array([float(x) for x in args.params.split(",")])
        print(f"Evaluating with parameters: {params}")

        # Evaluate and visualize
        results = identifier.evaluate_and_visualize(params)
    else:
        # Run the optimization
        options = {"maxiter": args.max_iter, "disp": True}
        result = identifier.optimize(method=args.method, options=options)

        print("\nOptimization Results:")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Final error: {result.fun}")
        print(f"Optimized parameters: {result.x}")

        # Save optimization results
        np.save(
            os.path.join(args.output_dir, "optimization_results.npy"),
            {
                "success": result.success,
                "iterations": result.nit,
                "function_evaluations": result.nfev,
                "final_error": result.fun,
                "optimized_parameters": result.x,
                "message": result.message,
            },
        )

        # Evaluate with the optimized parameters
        print("\nEvaluating with optimized parameters:")
        results = identifier.evaluate_and_visualize(result.x)

        # Save the optimized parameters in a format for later use
        np.save(
            os.path.join(args.output_dir, "optimized_parameters.npy"),
            {
                "joint_damping": results["optimized_damping"],
                "gain_scaling": results["gain_scaling"],
                "rmse_position": results["rmse_pos"],
                "rmse_orientation": results["rmse_ori"],
            },
        )

        # For easier reuse, also save as text
        with open(os.path.join(args.output_dir, "optimized_parameters.txt"), "w") as f:
            f.write("# Optimized parameters for Franka robot\n")
            f.write(f"# Position RMSE: {results['rmse_pos']:.4f} meters\n")
            f.write(f"# Orientation RMSE: {results['rmse_ori']:.4f} radians\n\n")

            f.write("# Joint damping values\n")
            for i, value in enumerate(results["optimized_damping"]):
                f.write(f"joint_damping_{i}: {value}\n")

            f.write(f"\n# Controller gain scaling\n")
            f.write(f"gain_scaling: {results['gain_scaling']}\n")

    print("\nSystem identification complete!")


if __name__ == "__main__":
    main()
