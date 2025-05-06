#!/usr/bin/env python3
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import modules from the parent directory
import numpy as np
import mujoco
import time
import jax.numpy as jnp
import roslibpy
from scipy.spatial.transform import Rotation

from hydrax.tasks.franka_push import FrankaPush
from hydrax_hardware_interface import HydraxHardwareInterface
from franka import FrankaRosInterface


class FrankaPushHydraxController(HydraxHardwareInterface):
    """Franka-specific implementation of Hydrax hardware interface for the push task"""

    def __init__(
        self,
        controller_type="cem",
        controller_config=None,
        control_frequency=100.0,
        planning_frequency=10.0,
        ros_host="localhost",
        ros_port=9090,
        initial_knots=None,
    ):
        # Create Franka push task
        task = FrankaPush()

        # Initialize ROS interface for Franka robot control
        self.franka = FrankaRosInterface(host=ros_host, port=ros_port)

        # Initialize ROS client - reuse the one from FrankaRosInterface
        self.client = self.franka.client

        # Subscribe to box pose
        self.box_pose = None
        self.box_pose_sub = roslibpy.Topic(
            self.client, "/box/pose", "geometry_msgs/PoseStamped"
        )
        self.box_pose_sub.subscribe(self.box_pose_callback)

        # Initialize base class
        super().__init__(
            task=task,
            controller_type=controller_type,
            controller_config=controller_config,
            control_frequency=control_frequency,
            planning_frequency=planning_frequency,
            initial_knots=initial_knots,
        )

        # Initialize the goal (target pose for the box)
        self.target_position = np.array([0.5, 0.0, 0.4])
        self.target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion w,x,y,z

    def box_pose_callback(self, message):
        """Callback function for receiving box pose updates using roslibpy"""
        # Store the box pose for later use in update_state
        # With roslibpy, we get a dictionary instead of an object with attributes
        self.box_pose = message.get("pose", {})

    def update_state(self):
        """Update internal MuJoCo state with Franka and box state data"""
        # Update MuJoCo data with latest robot state
        self.mj_data.time = time.time()
        self.mj_data.qpos[:7] = self.franka.q
        self.mj_data.qvel[:7] = self.franka.dq

        # Set the mocap position to the target position (where we want the box to go)
        self.mj_data.mocap_pos[0] = self.target_position
        self.mj_data.mocap_quat[0] = self.target_orientation

        # Update box position and orientation if we have received a box pose
        if self.box_pose is not None:
            # Extract position from pose message
            box_pos = np.array(
                [
                    self.box_pose["position"]["x"],
                    self.box_pose["position"]["y"],
                    self.box_pose["position"]["z"],
                ]
            )

            # Extract quaternion from pose message
            box_quat = np.array(
                [
                    self.box_pose["orientation"]["w"],
                    self.box_pose["orientation"]["x"],
                    self.box_pose["orientation"]["y"],
                    self.box_pose["orientation"]["z"],
                ]
            )

            # Find the box body in the model - first get the body ID
            box_body_id = mujoco.mj_name2id(
                self.task.mj_model, mujoco.mjtObj.mjOBJ_BODY, "box"
            )
            if box_body_id >= 0:
                # Set the box position in the qpos array
                # The position starts at the beginning of the box body's qpos segment
                # For a free joint (7 DoF), this would be [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
                box_qpos_adr = self.task.mj_model.body_jntadr[box_body_id]
                self.mj_data.qpos[box_qpos_adr : box_qpos_adr + 3] = box_pos
                self.mj_data.qpos[box_qpos_adr + 3 : box_qpos_adr + 7] = box_quat

        # Run forward to update all derived quantities
        mujoco.mj_forward(self.task.mj_model, self.mj_data)

    def send_command(self, action):
        """Send control action to the Franka robot"""
        # Extract cartesian position and orientation commands
        position = action[:3]
        orientation = action[3:]

        # Send to the ROS interface
        self.franka.send_cartesian_command(position, orientation)

    def set_goal(self, goal_position, goal_orientation=None):
        """Set a new target position and orientation for the box"""
        self.target_position = np.array(goal_position)
        if goal_orientation is not None:
            self.target_orientation = np.array(goal_orientation)

    def stop(self):
        """Stop all running threads and clean up resources"""
        # Unsubscribe from topics before stopping
        self.box_pose_sub.unsubscribe()

        # Call the parent class's stop method
        super().stop()


def main():
    """Example usage of the Franka Push Hydrax Controller"""
    import argparse

    parser = argparse.ArgumentParser(description="Franka Push Control using Hydrax")
    parser.add_argument(
        "--host", type=str, default="localhost", help="ROS websocket host"
    )
    parser.add_argument("--port", type=int, default=9090, help="ROS websocket port")
    parser.add_argument(
        "--controller",
        type=str,
        default="cem",
        choices=["ps", "mppi", "cem"],
        help="Controller type",
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Control duration in seconds"
    )
    parser.add_argument("--debug", action="store_true", help="Start debug viewer")

    args = parser.parse_args()

    # Configure controller
    controller_config = {
        "num_samples": 128,
        "plan_horizon": 0.5,
        "num_knots": 6,
        "sigma_start": 0.1,
        "sigma_min": 0.005,
        "num_elites": 32,
        "spline_type": "zero",
    }

    # Initial knots configuration for the controller
    initial_knots = jnp.tile(
        jnp.array(
            [
                0.5,
                0.0,
                0.4,
                -3.14,
                0.0,
                0.0,
            ]
        ),
        (controller_config["num_knots"], 1),
    )

    # Create controller with built-in ROS interface
    controller = FrankaPushHydraxController(
        controller_type=args.controller,
        controller_config=controller_config,
        control_frequency=50.0,
        planning_frequency=5.0,
        ros_host=args.host,
        ros_port=args.port,
        initial_knots=initial_knots,
    )

    # Set the initial goal
    controller.set_goal([0.5, 0.0, 0.03])

    if args.debug:
        controller.start_debug_viewer(show_traces=True)

    # Start keyboard control thread for interactive goal setting
    keyboard_thread = start_keyboard_control(controller)
    print("Keyboard control enabled. Use the following keys:")
    print("  w/s: Move target along X axis (+/-)")
    print("  a/d: Move target along Y axis (+/-)")
    print("  q/e: Move target along Z axis (+/-)")
    print("  r: Reset to initial position [0.5, 0.0, 0.4]")
    print("  Esc: Exit")

    try:
        # Run control loop
        controller.run_control_loop(controller, duration=args.duration)
    finally:
        # Clean up
        controller.stop()
        controller.franka.close()
        # Signal keyboard thread to stop if it exists
        if keyboard_thread:
            keyboard_thread.join(timeout=1.0)


def start_keyboard_control(controller, step_size=0.05):
    """
    Start a thread that listens for keyboard input to control the target position

    Args:
        controller: FrankaPushHydraxController instance
        step_size: Step size for goal position adjustment (meters)

    Returns:
        Thread object for the keyboard control thread
    """
    import threading
    from pynput import keyboard

    def keyboard_control_thread():
        # Flag to indicate if the thread should terminate
        stop_thread = threading.Event()

        def on_press(key):
            try:
                # Get current goal position
                current_goal = controller.target_position.copy()

                # Convert key to string if possible
                k = key.char.lower() if hasattr(key, "char") else None

                # Update goal based on key press
                if k == "w":
                    current_goal[0] += step_size
                elif k == "s":
                    current_goal[0] -= step_size
                elif k == "d":
                    current_goal[1] += step_size
                elif k == "a":
                    current_goal[1] -= step_size
                elif k == "q":
                    current_goal[2] += step_size
                elif k == "e":
                    current_goal[2] -= step_size
                elif k == "r":
                    current_goal = np.array([0.5, 0.0, 0.4])
                elif key == keyboard.Key.esc:
                    print("Exiting keyboard control")
                    stop_thread.set()
                    return False
                else:
                    return

                # Set the new goal position
                controller.set_goal(current_goal)
                print(f"New target: {current_goal}")
            except Exception as e:
                print(f"Error in keyboard handler: {e}")

        # Create keyboard listener
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        # Wait until stop flag is set
        try:
            while not stop_thread.is_set():
                time.sleep(0.1)
        finally:
            listener.stop()

    # Create and start the thread
    thread = threading.Thread(target=keyboard_control_thread, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    main()
