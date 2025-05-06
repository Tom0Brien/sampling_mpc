#!/usr/bin/env python3
import numpy as np
import mujoco
from hydrax.tasks.franka_reach import FrankaReach
import time

from hydrax_hardware_interface import HydraxHardwareInterface, ControlResult
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

    def send_command(self, action):
        """Send control action to the Franka robot"""
        # Delegate to the ROS interface
        print(f"Sending command: {action}")
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

    args = parser.parse_args()

    # Configure controller
    controller_config = {
        "num_samples": 512,
        "plan_horizon": 0.5,
        "num_knots": 11,
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
        control_frequency=100.0,
        planning_frequency=5.0,
        ros_host=args.host,
        ros_port=args.port,
        initial_knots=initial_knots,
    )

    # Set a goal
    controller.set_goal([0.5, 0.2, 0.5])

    try:
        # Run control loop
        # Note: hardware_interface is now the controller itself, since it has the franka interface
        controller.run_control_loop(controller, duration=args.duration)
    finally:
        # Clean up
        controller.stop()
        controller.franka.close()


if __name__ == "__main__":
    main()
