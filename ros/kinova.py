#!/usr/bin/env python3
import roslibpy
import time
import numpy as np


class KinovaRosInterface:
    """
    Interface for controlling Kinova Gen3 robot using ROS for communication.
    This class handles the ROS connection, state tracking, and sending commands.
    """

    def __init__(self, host="localhost", port=9090, robot_name="my_gen3"):
        # Robot namespace
        self.robot_name = robot_name

        # ROS connection
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        print(f"Connected to ROS: {self.client.is_connected}")

        # Topic names
        base_topic = f"/{self.robot_name}"
        self.joint_state_topic = f"{base_topic}/joint_states"
        self.base_feedback_topic = f"{base_topic}/base_feedback"

        # Setup subscribers for robot state
        self.joint_state_sub = roslibpy.Topic(
            self.client, self.joint_state_topic, "sensor_msgs/JointState"
        )

        self.base_feedback_sub = roslibpy.Topic(
            self.client, self.base_feedback_topic, "kortex_driver/BaseCyclic_Feedback"
        )

        # Add the play_joint_trajectory service
        self.play_joint_trajectory_service = roslibpy.Service(
            self.client,
            f"{base_topic}/base/play_joint_trajectory",
            "kortex_driver/PlayJointTrajectory",
        )

        # Add the play_cartesian_trajectory service
        self.play_cartesian_trajectory_service = roslibpy.Service(
            self.client,
            f"{base_topic}/base/play_cartesian_trajectory",
            "kortex_driver/PlayCartesianTrajectory",
        )

        # State information
        self.joint_positions = None
        self.ee_position = np.zeros(3)
        self.ee_orientation = np.zeros(3)

        # Subscribe to joint states
        self.joint_state_sub.subscribe(self.joint_state_callback)

        # Subscribe to base feedback
        self.base_feedback_sub.subscribe(self.base_feedback_callback)

        # Wait for initial state information
        print("Waiting for joint state and feedback messages...")
        timeout = 10.0  # seconds
        start_time = time.time()
        while (
            self.joint_positions is None or self.ee_position is None
        ) and time.time() - start_time < timeout:
            time.sleep(0.1)

        if self.joint_positions is None or self.ee_position is None:
            print("Warning: Did not receive initial state information within timeout")
        else:
            print("Received initial state information")

    def joint_state_callback(self, message):
        """Callback for joint state messages"""
        if "position" in message:
            self.joint_positions = np.array(message["position"])
            print(f"Joint positions: {self.joint_positions}")

    def base_feedback_callback(self, message):
        """Callback for base feedback messages"""
        # print(f"Base feedback: {message}")
        if "base" in message:
            print(f"Base feedback: {message['base']}")
            base = message["base"]
            # Extract end-effector position and orientation
            self.ee_position = np.array(
                [base["tool_pose_x"], base["tool_pose_y"], base["tool_pose_z"]]
            )
            print(f"EE position: {self.ee_position}")
            self.ee_orientation = np.array(
                [
                    base["tool_pose_theta_x"],
                    base["tool_pose_theta_y"],
                    base["tool_pose_theta_z"],
                ]
            )
            print(f"EE orientation: {self.ee_orientation}")

    def send_joint_trajectory(self, joint_positions):
        """
        Send a joint trajectory command using the play_joint_trajectory service
        joint_positions: List of 7 joint angles in radians
        """
        # Convert to degrees
        joint_positions = np.rad2deg(joint_positions)
        print(f"Sending joint trajectory: {joint_positions}")

        # Check if the joint_positions list is empty
        if len(joint_positions) == 0:
            print("Joint positions list is empty")
            return False

        if len(joint_positions) != 7:
            print(f"Expected 7 joint positions, got {len(joint_positions)}")
            return False

        # Create joint angle list
        joint_angles = []
        for i, angle in enumerate(joint_positions):
            joint_angles.append({"joint_identifier": i, "value": float(angle)})

        # Create the proper message structure
        # The kortex_driver/PlayJointTrajectoryRequest expects a kortex_driver/JointAngles object
        # not a list directly
        request = {
            "input": {
                "joint_angles": {"joint_angles": joint_angles},
                "constraint": {
                    "type": 0,  # default
                    "value": 0,  # default
                },
            }
        }

        try:
            _ = self.play_joint_trajectory_service.call(request)
            print("Joint trajectory command sent successfully")
            return True
        except Exception as e:
            print(f"Failed to send joint trajectory command: {str(e)}")
            return False

    def send_cartesian_trajectory(self, cartesian_pose):
        """
        Send a cartesian trajectory command using the play_cartesian_trajectory service
        cartesian_pose: List of 6 cartesian pose values [x, y, z, theta_x, theta_y, theta_z]
        """
        # Convert to orientation component to degrees
        cartesian_pose[3:] = np.rad2deg(cartesian_pose[3:])
        print(f"Sending cartesian trajectory: {cartesian_pose}")

        # Create the proper message structure
        # The kortex_driver/PlayCartesianTrajectoryRequest expects a kortex_driver/CartesianTrajectory object
        # not a list directly
        request = {
            "input": {
                "target_pose": {
                    "x": float(cartesian_pose[0]),
                    "y": float(cartesian_pose[1]),
                    "z": float(cartesian_pose[2]),
                    "theta_x": float(cartesian_pose[3]),
                    "theta_y": float(cartesian_pose[4]),
                    "theta_z": float(cartesian_pose[5]),
                }
            }
        }

        try:
            _ = self.play_cartesian_trajectory_service.call(request)
            print("Cartesian trajectory command sent successfully")
            return True
        except Exception as e:
            print(f"Failed to send cartesian trajectory command: {str(e)}")
            return False

    def close(self):
        """Close all connections to the robot"""
        if hasattr(self, "joint_state_sub") and self.joint_state_sub:
            self.joint_state_sub.unsubscribe()

        if hasattr(self, "base_feedback_sub") and self.base_feedback_sub:
            self.base_feedback_sub.unsubscribe()

        if hasattr(self, "client") and self.client.is_connected:
            self.client.terminate()

        print("ROS connections closed")
