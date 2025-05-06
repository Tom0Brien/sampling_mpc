#!/usr/bin/env python3
import numpy as np
import mujoco
from hydrax.tasks.franka_reach import FrankaReach
import time

from hydrax_hardware_interface import HydraxHardwareInterface
from franka import FrankaRosInterface
import jax.numpy as jnp


class FrankaHydraxController(HydraxHardwareInterface):
    """Franka-specific implementation of Hydrax hardware interface"""

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
        # Create Franka reach task
        task = FrankaReach()

        # Initialize ROS interface, will receive joint states and send cartesian commands
        self.franka = FrankaRosInterface(host=ros_host, port=ros_port)

        # Initialize base class
        super().__init__(
            task=task,
            controller_type=controller_type,
            controller_config=controller_config,
            control_frequency=control_frequency,
            planning_frequency=planning_frequency,
            initial_knots=initial_knots,
        )

        # Initialize the goal
        self.target_position = np.array([0.5, 0.0, 0.4])

    def update_state(self):
        """Update internal MuJoCo state with Franka state data and current time"""

        # Update MuJoCo data with latest robot state
        self.mj_data.time = time.time()
        self.mj_data.qpos[:7] = self.franka.q
        self.mj_data.qvel[:7] = self.franka.dq

        # Set the mocap position to the target position
        self.mj_data.mocap_pos[0] = self.target_position
        mujoco.mj_forward(self.task.mj_model, self.mj_data)

    def send_command(self, action):
        """Send control action to the Franka robot"""
        # Delegate to the ROS interface
        self.franka.send_cartesian_command(action[:3], action[3:])

    def set_goal(self, goal_position):
        """Set a new target position for the controller"""
        self.target_position = np.array(goal_position)


def main():
    """Example usage of the Franka Hydrax Controller"""
    import argparse

    parser = argparse.ArgumentParser(description="Franka Control using Hydrax")
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
        "--duration", type=float, default=60.0, help="Control duration in seconds"
    )
    parser.add_argument("--debug", action="store_true", help="Start debug viewer")

    args = parser.parse_args()

    # Configure controller
    controller_config = {
        "num_samples": 1024,
        "plan_horizon": 0.5,
        "num_knots": 6,
        "sigma_start": 0.1,
        "sigma_min": 0.005,
        "num_elites": 32,
        "spline_type": "zero",
    }

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
    controller = FrankaHydraxController(
        controller_type=args.controller,
        controller_config=controller_config,
        control_frequency=50.0,
        planning_frequency=5.0,
        ros_host=args.host,
        ros_port=args.port,
        initial_knots=initial_knots,
    )

    # Set the initial goal
    controller.set_goal([0.5, 0.0, 0.4])

    if args.debug:
        controller.start_debug_viewer()

    # Start keyboard control thread
    keyboard_thread = start_keyboard_control(controller)
    print("Keyboard control enabled. Use the following keys:")
    print("  w/s: Move along X axis (+/-)")
    print("  a/d: Move along Y axis (+/-)")
    print("  q/e: Move along Z axis (+/-)")
    print("  r: Reset to initial position [0.5, 0.0, 0.4]")
    print("  Esc: Exit")

    try:
        # Run control loop
        # Note: hardware_interface is now the controller itself, since it has the franka interface
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
    Start a thread that listens for keyboard input using pynput

    Args:
        controller: FrankaHydraxController instance
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
                print(f"New goal: {current_goal}")
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
