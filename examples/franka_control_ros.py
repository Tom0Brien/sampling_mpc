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


from hydrax.algs import MPPI, PredictiveSampling
from hydrax.tasks.franka_reach import FrankaReach
from hydrax.tasks.franka_reach import GainOptimizationMode

"""
Control loop for Franka robot using ROS for communication and Hydrax for control.
This script connects to ROS, subscribes to the robot state topic, computes control
actions using a sampling-based controller, and sends commands to the robot.
"""


class FrankaRosControl:
    def __init__(
        self, host="localhost", port=9090, controller_type="ps", gain_mode="none"
    ):
        # ROS connection
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        print(f"Connected to ROS: {self.client.is_connected}")

        # Setup subscribers for robot state
        self.joint_state_sub = roslibpy.Topic(
            self.client, "/joint_states", "sensor_msgs/JointState"
        )

        # Setup publisher for cartesian pose commands
        self.cartesian_pose_pub = roslibpy.Topic(
            self.client,
            "/cartesian_impedance_example_controller/equilibrium_pose",
            "geometry_msgs/PoseStamped",
            queue_size=10,
        )
        self.cartesian_pose_pub.advertise()

        # Replace the publisher with a service client for dynamic reconfigure
        self.compliance_param_client = roslibpy.Service(
            self.client,
            "/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node/set_parameters",
            "dynamic_reconfigure/Reconfigure",
        )

        # Initial stiffness/damping values (similar to C++ example defaults)
        self.translational_stiffness = 50.0  # N/m
        self.rotational_stiffness = 20.0  # Nm/rad
        self.nullspace_stiffness = 1.0

        # Send initial compliance parameters
        self.update_compliance_params(
            self.translational_stiffness,
            self.rotational_stiffness,
            self.nullspace_stiffness,
        )

        # Wait for publishers to be ready
        time.sleep(1.0)

        # Latest state information
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_end_effector_pose = None

        # Map gain_mode string to GainOptimizationMode enum
        gain_mode_map = {
            "none": GainOptimizationMode.NONE,
            "individual": GainOptimizationMode.INDIVIDUAL,
            "simple": GainOptimizationMode.SIMPLE,
        }
        gain_mode_enum = gain_mode_map.get(gain_mode, GainOptimizationMode.NONE)

        # Initialize the Hydrax task and controller
        self.task = FrankaReach(gain_mode=gain_mode_enum)
        self.mj_model = self.task.mj_model
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # Set the initial joint positions
        self.mj_data.qpos[:7] = [-0.196, -0.189, 0.182, -2.1, 0.0378, 1.91, 0.756]

        # Initial desired pose as a single 7D vector: [x, y, z, qx, qy, qz, qw]
        self.desired_pose = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])

        # Set up the controller based on type
        if controller_type == "ps":
            print("Using Predictive Sampling controller")
            self.controller = PredictiveSampling(
                self.task,
                num_samples=128,
                noise_level=0.05,
            )
        elif controller_type == "mppi":
            print("Using MPPI controller")
            noise_level = 0.01
            if gain_mode_enum == GainOptimizationMode.SIMPLE:
                # Set noise levels for position, orientation, and gains
                noise_level = np.array(
                    [0.01] * 6 + [1] + [1]
                )  # 6 pose params + 2 gain params

            self.controller = MPPI(
                self.task,
                num_samples=2000,
                noise_level=noise_level,
                temperature=0.001,
            )
        else:
            raise ValueError(f"Unsupported controller type: {controller_type}")

        # JIT compile the optimizer
        self.jit_optimize = jax.jit(self.controller.optimize, donate_argnums=(1,))
        self.get_action = jax.jit(self.controller.get_action)

        # Initialize policy parameters
        self.policy_params = self.controller.init_params()

        # Set up joint state callback
        self.joint_state_sub.subscribe(self.joint_state_callback)

        # Wait for first joint state message
        print("Waiting for joint state messages...")
        while self.current_joint_positions is None:
            time.sleep(0.1)
        print("Received initial joint state")

        # Add logger subscription
        self.rosout_sub = roslibpy.Topic(self.client, "/rosout", "rosgraph_msgs/Log")
        self.rosout_sub.subscribe(self.rosout_callback)

    def joint_state_callback(self, message):
        """Callback for joint state messages"""
        # Extract joint positions and velocities
        self.current_joint_positions = np.array(message["position"])
        self.current_joint_velocities = np.array(message["velocity"])

    def update_controller_state(self):
        """Update the controller's state with latest robot state"""

        # Update the state in mjx_data (need to map joint states to qpos/qvel)
        self.mjx_data = self.mjx_data.replace(
            qpos=jnp.array(self.current_joint_positions),
            qvel=jnp.array(self.current_joint_velocities),
        )
        # Store a CPU version of the state for visualization and transforms
        self.mj_data.qpos = np.array(self.current_joint_positions)
        self.mj_data.qvel = np.array(self.current_joint_velocities)
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

    def send_cartesian_command(self, pose):
        """
        Send a cartesian pose command to the robot
        pose: [x, y, z, roll, pitch, yaw]
        """
        # Transform the pose to the robot base frame (mujoco control is in reference frame)
        Rbr = self.mj_data.site_xmat[self.controller.task.reference_id].reshape(3, 3)
        rRBb = self.mj_data.site_xpos[self.controller.task.reference_id]

        rGBb = rRBb + Rbr @ pose[0:3]
        Rrg = Rotation.from_euler("xyz", pose[3:6])
        # Rotate the desired pose from reference frame to robot base frame
        Rbg = Rbr @ Rrg.as_matrix()
        r = Rotation.from_matrix(Rbg)
        quat = r.as_quat()  # [x, y, z, w]
        # Normalize the quaternion
        quat = quat / jnp.linalg.norm(quat)

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
                    "x": float(rGBb[0]),
                    "y": float(rGBb[1]),
                    "z": float(rGBb[2]),
                },
                "orientation": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                },
            },
        }

        # Save the desired pose as a single 7D vector: [x, y, z, qx, qy, qz, qw]
        self.desired_pose = np.concatenate([rGBb, quat])

        # Publish the pose message
        self.cartesian_pose_pub.publish(roslibpy.Message(pose_msg))

    def rosout_callback(self, message):
        # Filter by log level if desired (1=DEBUG, 2=INFO, 4=WARN, 8=ERROR, 16=FATAL)
        if message.get("level", 0) >= 4:  # WARN or higher
            print(
                f"ROS LOG [{message.get('name', 'unknown')}]: {message.get('msg', '')}"
            )

    def run_control_loop(
        self, frequency=10, duration=None, enable_viewer=False, fixed_camera_id=None
    ):
        """
        Run the control loop at the specified frequency
        frequency: control frequency in Hz
        duration: duration in seconds (None for infinite)
        enable_viewer: whether to enable the MuJoCo viewer for debugging
        fixed_camera_id: camera ID to use for the fixed camera view
        """
        period = 1.0 / frequency
        start_time = time.time()
        iteration = 0

        # For tracking costs and gains over time (added from deterministic.py)
        cost_history = []
        p_gain_history = []
        d_gain_history = []
        control_history = []

        # For visualization of rollouts
        max_traces = 5
        trace_width = 5.0
        trace_color = [1.0, 1.0, 1.0, 0.1]

        print(f"Starting control loop at {frequency} Hz")

        viewer = None
        try:
            # Initialize the MuJoCo viewer if requested
            if enable_viewer:
                print("Starting MuJoCo viewer for debugging")
                viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
                if fixed_camera_id is not None:
                    # Set the custom camera
                    viewer.cam.fixedcamid = fixed_camera_id
                    viewer.cam.type = 2

                # Set up rollout traces
                if hasattr(self.controller.task, "trace_site_ids"):
                    num_trace_sites = len(self.controller.task.trace_site_ids)
                    num_traces = min(self.controller.task.planning_horizon, max_traces)

                    for i in range(
                        num_trace_sites
                        * num_traces
                        * self.controller.task.planning_horizon
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
                if self.task.gain_mode == GainOptimizationMode.SIMPLE:
                    # Extract control and gains
                    pose_command, p_gains, d_gains = self.task.extract_gains(action)
                    print(f"Sending pose: {pose_command}")
                    self.send_cartesian_command(pose_command)

                    # Update the compliance parameters
                    trans_p_gain = action[self.task.nu_ctrl]
                    rot_p_gain = action[self.task.nu_ctrl + 1]
                    self.update_compliance_params(trans_p_gain, rot_p_gain)

                    # Store history
                    control_history.append(np.array(pose_command))
                    p_gain_history.append(np.array(p_gains))
                    d_gain_history.append(np.array(d_gains))
                else:
                    # Action is directly the pose command
                    pose_command = action
                    print(f"Sending pose: {pose_command}")
                    self.send_cartesian_command(pose_command)

                    # Store control history
                    control_history.append(np.array(pose_command))

                # Update the viewer if enabled
                if enable_viewer:
                    # Clear previous geoms
                    viewer.user_scn.ngeom = 0

                    # Visualize the rollouts
                    if hasattr(self.controller.task, "trace_site_ids"):
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

                    # Position for cost text - above the scene
                    cost_pos = np.array([0.0, 0.0, 0.5])

                    # Add cost text
                    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=np.array([0.2, 0.2, 0.2]),
                        pos=cost_pos,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 1, 1, 1]),  # white text
                    )
                    geom.label = f"Cost: {total_cost:.4f}"
                    viewer.user_scn.ngeom += 1

                    # Add gain visualization if in simple gain mode
                    if self.task.gain_mode == GainOptimizationMode.SIMPLE:
                        # Format gain text
                        trans_p_gain = action[self.task.nu_ctrl]
                        rot_p_gain = action[self.task.nu_ctrl + 1]
                        trans_d_gain = 2.0 * np.sqrt(trans_p_gain)
                        rot_d_gain = 2.0 * np.sqrt(rot_p_gain)

                        p_gain_text = f"Trans P-gain: {trans_p_gain:.2f}, Rot P-gain: {rot_p_gain:.2f}"
                        d_gain_text = f"Trans D-gain: {trans_d_gain:.2f}, Rot D-gain: {rot_d_gain:.2f}"

                        # Add P-gain text (positioned below cost text)
                        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                        mujoco.mjv_initGeom(
                            geom,
                            type=mujoco.mjtGeom.mjGEOM_LABEL,
                            size=np.array([0.15, 0.15, 0.15]),
                            pos=cost_pos
                            - np.array([0.0, 0.0, 0.05]),  # position below cost
                            mat=np.eye(3).flatten(),
                            rgba=np.array([0.8, 0.8, 1.0, 1]),  # light blue text
                        )
                        geom.label = p_gain_text
                        viewer.user_scn.ngeom += 1

                        # Add D-gain text (positioned below P-gain text)
                        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                        mujoco.mjv_initGeom(
                            geom,
                            type=mujoco.mjtGeom.mjGEOM_LABEL,
                            size=np.array([0.15, 0.15, 0.15]),
                            pos=cost_pos
                            - np.array([0.0, 0.0, 0.1]),  # position below P-gain
                            mat=np.eye(3).flatten(),
                            rgba=np.array([0.8, 1.0, 0.8, 1]),  # light green text
                        )
                        geom.label = d_gain_text
                        viewer.user_scn.ngeom += 1

                    # Add visualization of desired pose with a red sphere
                    if hasattr(self, "desired_pose"):
                        # The desired_pose is now a single 7D vector: [x, y, z, qx, qy, qz, qw]
                        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                        mujoco.mjv_initGeom(
                            geom,
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.02, 0, 0],  # Size of the sphere (radius)
                            pos=self.desired_pose[:3],  # First 3 elements are position
                            mat=np.eye(3).flatten(),
                            rgba=[
                                1.0,
                                0.0,
                                0.0,
                                0.8,
                            ],  # Red color with some transparency
                        )
                        viewer.user_scn.ngeom += 1

                        # Add a text label for the desired pose
                        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                        mujoco.mjv_initGeom(
                            geom,
                            type=mujoco.mjtGeom.mjGEOM_LABEL,
                            size=np.array([0.15, 0.15, 0.15]),
                            pos=self.desired_pose[:3]
                            + np.array([0.0, 0.0, 0.05]),  # Position above the sphere
                            mat=np.eye(3).flatten(),
                            rgba=np.array([1, 1, 1, 1]),  # White text
                        )
                        geom.label = "Target"
                        viewer.user_scn.ngeom += 1

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
            self.rosout_sub.unsubscribe()
            self.client.terminate()
            print("\nROS connections closed")

            # Plot cost and gain history
            if cost_history and len(cost_history) > 1:
                try:
                    # Create figure with multiple subplots
                    fig, axes = plt.subplots(
                        1
                        + (2 if self.task.gain_mode != GainOptimizationMode.NONE else 0)
                        + 1,
                        1,
                        figsize=(10, 10),
                        sharex=True,
                    )

                    # If not optimizing gains, axes is not a list, so make it one for consistency
                    if self.task.gain_mode == GainOptimizationMode.NONE:
                        axes = [axes] if not isinstance(axes, np.ndarray) else axes

                    # Plot cost history
                    axes[0].plot(cost_history)
                    axes[0].set_title("Total Cost Over Time")
                    axes[0].set_ylabel("Cost")
                    axes[0].grid(True)

                    # Plot gain histories if optimizing gains
                    if (
                        self.task.gain_mode != GainOptimizationMode.NONE
                        and p_gain_history
                        and d_gain_history
                    ):
                        # Convert lists of arrays to 2D arrays
                        p_gains_array = np.array(p_gain_history)
                        d_gains_array = np.array(d_gain_history)

                        # Plot P gains
                        for i in range(p_gains_array.shape[1]):
                            axes[1].plot(p_gains_array[:, i], label=f"Actuator {i}")
                        axes[1].set_title("P Gains Over Time")
                        axes[1].set_ylabel("P Gain")
                        axes[1].grid(True)
                        axes[1].legend()

                        # Plot D gains
                        for i in range(d_gains_array.shape[1]):
                            axes[2].plot(d_gains_array[:, i], label=f"Actuator {i}")
                        axes[2].set_title("D Gains Over Time")
                        axes[2].set_ylabel("D Gain")
                        axes[2].grid(True)
                        axes[2].legend()

                        # Plot control signals (last subplot)
                        control_plot_idx = 3
                    else:
                        # If not optimizing gains, control plot is the second subplot
                        control_plot_idx = 1

                    # Plot control history
                    if control_history:
                        control_array = np.array(control_history)
                        for i in range(control_array.shape[1]):
                            axes[control_plot_idx].plot(
                                control_array[:, i], label=f"Control {i}"
                            )
                        axes[control_plot_idx].set_title("Control Signals Over Time")
                        axes[control_plot_idx].set_ylabel("Control Value")
                        axes[control_plot_idx].set_xlabel("Control Steps")
                        axes[control_plot_idx].grid(True)
                        axes[control_plot_idx].legend()

                    plt.tight_layout()
                    plt.savefig("recordings/ros_control_history.png")
                    plt.show()
                    print(
                        "Cost, gain, and control history plotted and saved to 'recordings/ros_control_history.png'"
                    )
                except ImportError:
                    print(
                        "Matplotlib not available for plotting. Install with 'pip install matplotlib'"
                    )
                except Exception as e:
                    print(f"Error plotting results: {str(e)}")


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
        choices=["ps", "mppi"],
        help="Controller type (ps for Predictive Sampling, mppi for MPPI)",
    )
    parser.add_argument(
        "--gain-mode",
        type=str,
        choices=["none", "individual", "simple"],
        default="none",
        help="Gain optimization mode (none, individual, or simple)",
    )
    parser.add_argument(
        "--enable-viewer",
        action="store_true",
        default=True,
        help="Enable MuJoCo viewer for debugging",
    )
    parser.add_argument(
        "--fixed-camera-id",
        type=int,
        default=None,
        help="Camera ID for fixed view in MuJoCo viewer",
    )

    args = parser.parse_args()

    controller = FrankaRosControl(
        host=args.host,
        port=args.port,
        controller_type=args.controller,
        gain_mode=args.gain_mode,
    )

    controller.run_control_loop(
        frequency=args.frequency,
        duration=args.duration,
        enable_viewer=args.enable_viewer,
        fixed_camera_id=args.fixed_camera_id,
    )


if __name__ == "__main__":
    print("Starting Franka control loop")
    main()
