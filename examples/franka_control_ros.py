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

from hydrax.algs import MPPI, PredictiveSampling
from hydrax.tasks.franka_reach import FrankaReach

"""
Control loop for Franka robot using ROS for communication and Hydrax for control.
This script connects to ROS, subscribes to the robot state topic, computes control
actions using a sampling-based controller, and sends commands to the robot.
"""


class FrankaRosControl:
    def __init__(
        self, host="localhost", port=9090, controller_type="ps", optimize_gains=False
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

        # Wait for publishers to be ready
        time.sleep(1.0)

        # Latest state information
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_end_effector_pose = None

        # Initialize the Hydrax task and controller
        self.task = FrankaReach(optimize_gains=optimize_gains)
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
            if optimize_gains:
                # Set noise levels for position, orientation, and gains
                noise_level = np.array(
                    [0.01] * 6 + [1.0] * 12
                )  # 6 pose params + 12 gain params

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
        # This depends on how your task is set up - you may need to adjust indices
        # TODO: Check this is correct
        self.mjx_data = self.mjx_data.replace(
            qpos=jnp.array(self.current_joint_positions),
            qvel=jnp.array(self.current_joint_velocities),
        )
        # Store a CPU version of the state for visualization and transforms
        self.mj_data.qpos = np.array(self.current_joint_positions)
        self.mj_data.qvel = np.array(self.current_joint_velocities)
        mujoco.mj_forward(self.mj_model, self.mj_data)

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

                # Handle the case where the action includes gains
                if self.task.optimize_gains:
                    # Extract control and gains
                    pose_command, p_gains, d_gains = self.task.extract_gains(action)
                    print(
                        f"Sending pose: {pose_command}, P-gains: {p_gains}, D-gains: {d_gains}"
                    )
                    self.send_cartesian_command(pose_command)
                    # Note: You might want to send the gains to a separate ROS topic if your
                    # controller supports dynamic gain adjustment
                else:
                    # Action is directly the pose command
                    pose_command = action
                    print(f"Sending pose: {pose_command}")
                    self.send_cartesian_command(pose_command)

                # Calculate cost for logging
                total_cost = float(jnp.sum(rollouts.costs[0]))

                # Update the viewer if enabled
                if enable_viewer:
                    # Clear previous geoms
                    viewer.user_scn.ngeom = 0

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
        "--optimize-gains",
        action="store_true",
        help="Optimize controller gains along with poses",
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
        optimize_gains=args.optimize_gains,
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
