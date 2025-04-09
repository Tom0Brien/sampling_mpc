#!/usr/bin/env python3
import roslibpy
import time
import numpy as np
import argparse
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt


from hydrax.algs import MPPI, PredictiveSampling, CEM
from hydrax import ROOT
from hydrax.tasks.particle3d import Particle3D
from hydrax.tasks.particle_collision import ParticleCollision
from hydrax.task_base import ControlMode
from parse_args import control_mode_map

"""
Control loop for Franka robot using ROS for communication and Hydrax for control.
This script connects to ROS, subscribes to the robot state topic, computes control
actions using a sampling-based controller, and sends commands to the robot.
"""


class FrankaRosControl:
    def __init__(
        self, host="localhost", port=9090, controller_type="ps", control_mode="general"
    ):
        # ROS connection
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        print(f"Connected to ROS: {self.client.is_connected}")

        # Setup subscribers for robot state
        self.joint_state_sub = roslibpy.Topic(
            self.client, "/joint_states", "sensor_msgs/JointState"
        )

        self.franka_state_sub = roslibpy.Topic(
            self.client,
            "/franka_state_controller/franka_states",
            "franka_msgs/FrankaState",
        )

        # Setup publisher for cartesian pose commands
        self.cartesian_pose_pub = roslibpy.Topic(
            self.client,
            "/cartesian_impedance_example_controller/equilibrium_pose",
            "geometry_msgs/PoseStamped",
            queue_size=10,
        )
        self.cartesian_pose_pub.advertise()

        # Setup dynamic reconfigure compliance service
        self.compliance_param_client = roslibpy.Service(
            self.client,
            "/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node/set_parameters",
            "dynamic_reconfigure/Reconfigure",
        )

        # Wait for publishers to be ready
        time.sleep(1.0)

        # Latest state information
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_end_effector_pose = None

        # Map control_mode string to ControlMode enum
        control_mode_enum = control_mode_map[control_mode]

        # Initialize the Hydrax task and controller
        # self.task = Particle3D(control_mode=control_mode_enum)
        self.task = ParticleCollision(control_mode=control_mode_enum)
        self.mj_model = self.task.mj_model
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # Add a franka model
        self.franka_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/franka_emika_panda/mjx_panda_cartesian.xml"
        )
        self.franka_data = mujoco.MjData(self.franka_model)
        # Set the initial joint positions - not needed for particle task
        # self.mj_data.qpos[:7] = [-0.196, -0.189, 0.182, -2.1, 0.0378, 1.91, 0.756]

        # Set up the controller based on type
        if controller_type == "ps":
            print("Using Predictive Sampling controller")
            self.controller = PredictiveSampling(
                self.task,
                num_samples=128,
                noise_level=0.01,
            )
        elif controller_type == "mppi":
            print("Using MPPI controller")
            self.controller = MPPI(
                self.task,
                num_samples=2000,
                noise_level=0.01,
                temperature=0.001,
            )
        if controller_type == "cem":
            print("Running CEM")
            self.controller = CEM(
                self.task,
                num_samples=512,
                num_elites=20,
                sigma_start=0.025,
                sigma_min=0.005,
                explore_fraction=0.5,
            )

        else:
            raise ValueError(f"Unsupported controller type: {controller_type}")

        # JIT compile the optimizer
        self.jit_optimize = jax.jit(self.controller.optimize, donate_argnums=(1,))
        self.get_action = jax.jit(self.controller.get_action)

        # Initial conditions
        self.translational_stiffness = 200.0  # N/m
        self.rotational_stiffness = 10.0  # Nm/rad
        self.nullspace_stiffness = 1.0
        initial_control = jnp.tile(
            jnp.array(
                [
                    0.5,  # x reference
                    0.0,  # y reference
                    0.5,  # z reference
                ]
            ),
            (self.task.planning_horizon, 1),
        )

        # Send initial compliance parameters
        self.update_compliance_params(
            self.translational_stiffness,
            self.rotational_stiffness,
            self.nullspace_stiffness,
        )

        # Initialize policy parameters and initial control sequence
        self.policy_params = self.controller.init_params(
            initial_control=initial_control
        )

        # Set up joint and franka state callbacks
        self.ee_position = np.zeros(3)
        self.ee_velocity = np.zeros(6)
        self.joint_state_sub.subscribe(self.joint_state_callback)
        self.franka_state_sub.subscribe(self.franka_state_callback)

        # Wait for first joint state message
        print("Waiting for joint state messages...")
        while self.current_joint_positions is None:
            time.sleep(0.1)
        print("Received initial joint state")
        self.update_controller_state()

        # Calculate control frequency parameters will be done in run_control_loop
        # since that's where we receive the requested frequency

    def joint_state_callback(self, message):
        """Callback for joint state messages"""
        # Extract joint positions and velocities
        self.current_joint_positions = np.array(message["position"])
        self.current_joint_velocities = np.array(message["velocity"])

    def franka_state_callback(self, message):
        """Callback for franka state messages"""
        # Extract franka state
        q = np.concatenate([np.array(message["q"]), np.zeros(2)])
        dq = np.concatenate([np.array(message["dq"]), np.zeros(2)])

        # Update franka model with latest joint positions and velocities
        self.franka_data.qpos = q
        self.franka_data.qvel = dq
        mujoco.mj_forward(self.franka_model, self.franka_data)

        # Get franka jacobian and compute ee velocity
        gripper_site_id = self.franka_model.site("gripper").id
        self.ee_position = self.franka_data.site_xpos[gripper_site_id]
        J = np.zeros((6, self.franka_model.nv))
        mujoco.mj_jacSite(
            self.franka_model, self.franka_data, J[:3, :], J[3:, :], gripper_site_id
        )
        self.ee_velocity = J @ dq

    def update_controller_state(self):
        """Update the controller's state with latest robot state"""
        # For particle task, we only need position and velocity
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        qpos[:3] = self.ee_position
        qvel = self.ee_velocity
        self.mjx_data = self.mjx_data.replace(
            qpos=qpos,
            qvel=qvel,
        )
        # Store a CPU version of the state for visualization and transforms
        self.mj_data.qpos = qpos
        self.mj_data.qvel = qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_compliance_params(
        self, translational_stiffness, rotational_stiffness, nullspace_stiffness=10.0
    ):
        """
        Send an update to the compliance parameters of the cartesian impedance controller
        using the dynamic reconfigure service
        """
        # Create a dynamic_reconfigure/Reconfigure service request
        request = {
            "config": {
                "bools": [],
                "ints": [],
                "strs": [],
                "doubles": [
                    {
                        "name": "translational_stiffness",
                        "value": float(translational_stiffness),
                    },
                    {
                        "name": "rotational_stiffness",
                        "value": float(rotational_stiffness),
                    },
                    {
                        "name": "nullspace_stiffness",
                        "value": float(nullspace_stiffness),
                    },
                ],
                "groups": [],
            }
        }

        print(
            f"Calling service to update compliance params: trans={translational_stiffness}, rot={rotational_stiffness}"
        )

        # Call the service
        self.compliance_param_client.call(request, self.compliance_service_callback)

    def compliance_service_callback(self, result):
        """Callback for the compliance parameter service call"""
        if "config" in result:
            print("Successfully updated compliance parameters")
            # You could parse and print the returned values here
        else:
            print("Failed to update compliance parameters:", result)

    def send_cartesian_command(self, position):
        """
        Send a cartesian pose command to the robot
        position: [x, y, z]
        """
        # Fixed orientation
        quat = Rotation.from_euler("xyz", [-3.14, 0.0, 0.0]).as_quat()

        # Create pose message
        pose_msg = {
            "header": {
                "frame_id": "panda_link0",
                "stamp": {
                    "secs": int(time.time()),
                    "nsecs": int((time.time() % 1) * 1e9),
                },
            },
            "pose": {
                "position": {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                },
                "orientation": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                },
            },
        }

        # Publish the pose message
        self.cartesian_pose_pub.publish(roslibpy.Message(pose_msg))

    def _init_viewer_traces(self, viewer, max_traces):
        """Initialize trace geometries for rollout visualization"""
        if not hasattr(self.controller.task, "trace_site_ids"):
            return

        num_trace_sites = len(self.controller.task.trace_site_ids)
        num_traces = min(self.controller.task.planning_horizon, max_traces)
        trace_color = [1.0, 1.0, 1.0, 0.1]

        for i in range(
            num_trace_sites * num_traces * self.controller.task.planning_horizon
        ):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.array(trace_color),
            )
            viewer.user_scn.ngeom += 1

    def _visualize_rollouts(self, viewer, rollouts, max_traces, trace_width):
        """Visualize controller rollouts"""
        if not hasattr(self.controller.task, "trace_site_ids"):
            return

        num_trace_sites = len(self.controller.task.trace_site_ids)
        num_traces = min(rollouts.controls.shape[1], max_traces)

        ii = 0
        for k in range(num_trace_sites):
            for i in range(num_traces):
                for j in range(self.controller.task.planning_horizon):
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[ii],
                        mujoco.mjtGeom.mjGEOM_LINE,
                        trace_width,
                        rollouts.trace_sites[i, j, k],
                        rollouts.trace_sites[i, j + 1, k],
                    )
                    ii += 1
                    viewer.user_scn.ngeom += 1

    def _add_cost_text(self, viewer, total_cost):
        """Add cost text to viewer"""
        cost_pos = np.array([0.0, 0.0, 0.5])
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LABEL,
            size=np.array([0.2, 0.2, 0.2]),
            pos=cost_pos,
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 1, 1, 1]),
        )
        geom.label = f"Cost: {total_cost:.4f}"
        viewer.user_scn.ngeom += 1

    def _visualize_desired_pose(self, viewer):
        """Visualize the desired poses with red spheres"""
        # Add sphere for desired pose
        for i in range(self.task.planning_horizon):
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                pos=self.policy_params.mean[i, :3],
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.0, 0.0, 0.8],
            )
            viewer.user_scn.ngeom += 1
            # Add label for desired pose
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_LABEL,
                size=np.array([0.15, 0.15, 0.15]),
                pos=self.policy_params.mean[i, :3] + np.array([0.0, 0.0, 0.05]),
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 1, 1, 1]),
            )
            geom.label = f"{i}"
            viewer.user_scn.ngeom += 1

    def _plot_histories(
        self, cost_history, p_gain_history, d_gain_history, control_history
    ):
        """Plot and save control histories

        Args:
            cost_history (list): History of cost values
            p_gain_history (list): History of P gains (optional)
            d_gain_history (list): History of D gains (optional)
            control_history (list): History of control signals
        """
        if not cost_history or len(cost_history) <= 1:
            return

        try:
            # Determine number of subplots needed
            num_subplots = 1  # Always have cost plot
            if self.task.control_mode == ControlMode.CARTESIAN_SIMPLE_VI:
                num_subplots += 2  # Add P and D gain plots
            num_subplots += 1  # Add control signals plot

            # Create figure
            fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 10), sharex=True)
            axes = [axes] if not isinstance(axes, np.ndarray) else axes

            # Plot cost history
            axes[0].plot(cost_history)
            axes[0].set_title("Total Cost Over Time")
            axes[0].set_ylabel("Cost")
            axes[0].grid(True)

            current_plot = 1

            # Plot gain histories if using CARTESIAN_SIMPLE_VI mode
            if (
                self.task.control_mode == ControlMode.CARTESIAN_SIMPLE_VI
                and p_gain_history
                and d_gain_history
            ):
                # Plot P gains
                p_gains_array = np.array(p_gain_history)
                for i in range(p_gains_array.shape[1]):
                    axes[current_plot].plot(p_gains_array[:, i], label=f"Actuator {i}")
                axes[current_plot].set_title("P Gains Over Time")
                axes[current_plot].set_ylabel("P Gain")
                axes[current_plot].grid(True)
                axes[current_plot].legend()
                current_plot += 1

                # Plot D gains
                d_gains_array = np.array(d_gain_history)
                for i in range(d_gains_array.shape[1]):
                    axes[current_plot].plot(d_gains_array[:, i], label=f"Actuator {i}")
                axes[current_plot].set_title("D Gains Over Time")
                axes[current_plot].set_ylabel("D Gain")
                axes[current_plot].grid(True)
                axes[current_plot].legend()
                current_plot += 1

            # Plot control signals
            if control_history:
                control_array = np.array(control_history)
                for i in range(control_array.shape[1]):
                    axes[current_plot].plot(control_array[:, i], label=f"Control {i}")
                axes[current_plot].set_title("Control Signals Over Time")
                axes[current_plot].set_ylabel("Control Value")
                axes[current_plot].set_xlabel("Control Steps")
                axes[current_plot].grid(True)
                axes[current_plot].legend()

            plt.tight_layout()
            plt.savefig("recordings/ros_control_history.png")
            plt.show()
            print(
                "Control histories plotted and saved to 'recordings/ros_control_history.png'"
            )

        except ImportError:
            print(
                "Matplotlib not available for plotting. Install with 'pip install matplotlib'"
            )
        except Exception as e:
            print(f"Error plotting results: {str(e)}")

    def run_control_loop(self, frequency=10, duration=None, enable_viewer=False):
        """
        Run the control loop at the specified frequency
        frequency: requested control frequency in Hz
        duration: duration in seconds (None for infinite)
        enable_viewer: whether to enable the MuJoCo viewer for debugging
        """
        # Calculate actual control frequency based on requested frequency and model timestep
        sim_steps_per_replan = max(
            int(1.0 / (frequency * self.mj_model.opt.timestep)), 1
        )
        step_dt = sim_steps_per_replan * self.mj_model.opt.timestep
        actual_frequency = 1.0 / step_dt
        print(
            f"Planning at {actual_frequency} Hz, "
            f"simulating at {1.0 / self.mj_model.opt.timestep} Hz"
        )

        period = step_dt
        start_time = time.time()
        iteration = 0

        # For tracking costs and gains over time
        cost_history = []
        p_gain_history = []
        d_gain_history = []
        control_history = []

        # For visualization of rollouts
        max_traces = 5

        # Keyboard control parameters
        keyboard_step_size = 0.01
        mocap_index = 0  # Index of mocap body to control

        print(f"Starting control loop at {actual_frequency} Hz")

        # Define keyboard callback
        def key_callback(keycode):
            key = keycode & 0xFF
            if key == 7:  # LEFT
                self.mj_data.mocap_pos[mocap_index, 0] -= keyboard_step_size
                self.mjx_data = self.mjx_data.replace(
                    mocap_pos=jnp.array(self.mj_data.mocap_pos)
                )
            elif key == 6:  # RIGHT
                self.mj_data.mocap_pos[mocap_index, 0] += keyboard_step_size
                self.mjx_data = self.mjx_data.replace(
                    mocap_pos=jnp.array(self.mj_data.mocap_pos)
                )
            elif key == 9:  # UP
                self.mj_data.mocap_pos[mocap_index, 1] += keyboard_step_size
                self.mjx_data = self.mjx_data.replace(
                    mocap_pos=jnp.array(self.mj_data.mocap_pos)
                )
            elif key == 8:  # DOWN
                self.mj_data.mocap_pos[mocap_index, 1] -= keyboard_step_size
                self.mjx_data = self.mjx_data.replace(
                    mocap_pos=jnp.array(self.mj_data.mocap_pos)
                )

        viewer = None
        try:
            # Initialize the MuJoCo viewer if requested
            if enable_viewer:
                print("Starting MuJoCo viewer for debugging")
                viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=key_callback,  # Add the keyboard callback
                )

                # Initialize trace geometries
                self._init_viewer_traces(viewer, max_traces)

            while self.client.is_connected:
                loop_start = time.time()

                # Check if duration is exceeded
                if duration is not None and time.time() - start_time > duration:
                    print(f"Control duration of {duration}s reached")
                    break

                # Check if viewer was closed
                if enable_viewer and not viewer.is_running():
                    print("Viewer closed, stopping control loop")
                    break

                # Update controller state with latest robot state
                self.update_controller_state()

                # Optimize control policy
                plan_start = time.time()
                self.policy_params, rollouts = self.jit_optimize(
                    self.mjx_data, self.policy_params
                )
                plan_time = time.time() - plan_start

                # Get control action
                t = 0.0  # Time within the current control interval
                action = self.get_action(self.policy_params, t)

                # Calculate cost for logging and history
                total_cost = float(jnp.sum(rollouts.costs[0]))
                cost_history.append(total_cost)

                # Handle the case where the action includes gains
                if self.task.control_mode == ControlMode.CARTESIAN_SIMPLE_VI:
                    # Extract control and gains
                    position_command, p_gains, d_gains = self.task.extract_gains(action)

                    # Update the compliance parameters
                    trans_p_gain = action[self.task.nu_ctrl]
                    rot_p_gain = action[self.task.nu_ctrl + 1]
                    self.update_compliance_params(trans_p_gain, rot_p_gain)

                    # Send cartesian pose command
                    self.send_cartesian_command(position_command)

                    # Store history
                    control_history.append(np.array(position_command))
                    p_gain_history.append(np.array(p_gains))
                    d_gain_history.append(np.array(d_gains))
                else:
                    # Action is directly the pose command
                    position_command = action
                    self.send_cartesian_command(position_command)

                    # Store control history
                    control_history.append(np.array(position_command))

                # Update the viewer if enabled
                if enable_viewer:
                    # Clear previous geoms
                    viewer.user_scn.ngeom = 0

                    # Update visualizations
                    self._visualize_rollouts(
                        viewer, rollouts, max_traces=5, trace_width=5.0
                    )
                    self._add_cost_text(viewer, total_cost)
                    self._visualize_desired_pose(viewer)

                    viewer.sync()

                # Sleep to maintain the desired control frequency
                elapsed = time.time() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)

                # Print status
                rtr = period / (time.time() - loop_start)  # Real-time ratio
                print(
                    f"Iter {iteration}: RTR: {rtr:.2f}, plan: {plan_time:.4f}s, cost: {total_cost:.4f}",
                    end="\r",
                )

                iteration += 1

        except KeyboardInterrupt:
            print("\nControl interrupted by user")
        finally:
            # Clean up MuJoCo viewer
            if enable_viewer and viewer is not None:
                viewer.close()

            # Clean up ROS connections
            self.cartesian_pose_pub.unadvertise()
            self.joint_state_sub.unsubscribe()
            self.client.terminate()
            print("\nROS connections closed")

            # Plot histories
            self._plot_histories(
                cost_history, p_gain_history, d_gain_history, control_history
            )


def main():
    parser = argparse.ArgumentParser(description="Franka Control using ROS and Hydrax")
    parser.add_argument(
        "--host", type=str, default="localhost", help="ROS websocket host"
    )
    parser.add_argument("--port", type=int, default=9090, help="ROS websocket port")
    parser.add_argument(
        "--frequency", type=float, default=10.0, help="Control frequency in Hz"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Control duration in seconds"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="ps",
        choices=["ps", "mppi", "cem"],
        help="Controller type (ps for Predictive Sampling, mppi for MPPI, cem for CEM)",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        choices=["general", "cartesian", "cartesian_simple_vi"],
        default="general",
        help="Control mode (general for general control, CARTESIAN_SIMPLE_VI for cartesian impedance control with variable gains)",
    )
    parser.add_argument(
        "--enable-viewer",
        action="store_true",
        default=False,
        help="Enable MuJoCo viewer for debugging",
    )

    args = parser.parse_args()

    controller = FrankaRosControl(
        host=args.host,
        port=args.port,
        controller_type=args.controller,
        control_mode=args.control_mode,
    )

    controller.run_control_loop(
        frequency=args.frequency,
        duration=args.duration,
        enable_viewer=args.enable_viewer,
    )


if __name__ == "__main__":
    print("Starting Franka control loop")
    main()
