#!/usr/bin/env python3
import roslibpy
import time
import numpy as np
import threading


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

        # Add action notification topic
        self.action_topic = f"{base_topic}/action_topic"
        self.action_notif_sub = roslibpy.Topic(
            self.client, self.action_topic, "kortex_driver/ActionNotification"
        )

        # Action tracking
        self.last_action_notif_type = None
        self.action_complete_event = threading.Event()
        self.action_complete_event.set()  # Initially set to True (no action in progress)

        # State information
        self.joint_positions = None
        self.ee_position = np.zeros(3)
        self.ee_orientation = np.zeros(3)

        # Profiling information
        self.movement_start_time = None
        self.movement_end_time = None
        self.last_movement_duration = None

        # Subscribe to joint states
        self.joint_state_sub.subscribe(self.joint_state_callback)

        # Subscribe to base feedback
        self.base_feedback_sub.subscribe(self.base_feedback_callback)

        # Subscribe to action notifications
        self.action_notif_sub.subscribe(self.action_notification_callback)

        # Activate action notifications
        self.activate_action_notifications()

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

    def base_feedback_callback(self, message):
        """Callback for base feedback messages"""
        if "base" in message:
            base = message["base"]
            # Extract end-effector position and orientation
            self.ee_position = np.array(
                [base["tool_pose_x"], base["tool_pose_y"], base["tool_pose_z"]]
            )
            self.ee_orientation = np.array(
                [
                    base["tool_pose_theta_x"],
                    base["tool_pose_theta_y"],
                    base["tool_pose_theta_z"],
                ]
            )

    def action_notification_callback(self, message):
        """Callback for action notification messages"""
        if "action_event" in message:
            self.last_action_notif_type = message["action_event"]

            # If the action is complete (either successfully or aborted)
            if self.last_action_notif_type in [1, 2]:  # ACTION_END=1, ACTION_ABORT=2
                # Record end time for profiling
                if self.movement_start_time is not None:
                    self.movement_end_time = time.time()
                    self.last_movement_duration = (
                        self.movement_end_time - self.movement_start_time
                    )
                    print(
                        f"Movement completed in {self.last_movement_duration:.3f} seconds"
                    )

                self.action_complete_event.set()
                if self.last_action_notif_type == 1:
                    print("Action completed successfully")
                else:
                    print("Action aborted")

    def activate_action_notifications(self):
        """Activate publishing of action notifications"""
        try:
            service = roslibpy.Service(
                self.client,
                f"/{self.robot_name}/base/activate_publishing_of_action_topic",
                "kortex_driver/OnNotificationActionTopic",
            )
            response = service.call(roslibpy.ServiceRequest({}))
            print("Action notifications activated")
            return True
        except Exception as e:
            print(f"Failed to activate action notifications: {str(e)}")
            return False

    def wait_for_action_completion(self, timeout=30.0):
        """Wait for the current action to complete"""
        return self.action_complete_event.wait(timeout)

    def send_joint_trajectory(self, joint_positions, wait_for_completion=True):
        """
        Send a joint trajectory command using the play_joint_trajectory service
        joint_positions: List of 7 joint angles in radians
        wait_for_completion: If True, wait for the action to complete before returning
        """
        # Wait for any previous action to complete
        if not self.action_complete_event.is_set():
            print("Waiting for previous action to complete...")
            if not self.wait_for_action_completion():
                print("Timed out waiting for previous action")
                return False

        # Reset completion event
        self.action_complete_event.clear()

        # Start time for profiling
        self.movement_start_time = time.time()

        # Convert to degrees
        joint_positions = np.rad2deg(joint_positions)
        print(f"Sending joint trajectory: {joint_positions}")

        # Check if the joint_positions list is empty
        if len(joint_positions) == 0:
            print("Joint positions list is empty")
            self.action_complete_event.set()
            return False

        if len(joint_positions) != 7:
            print(f"Expected 7 joint positions, got {len(joint_positions)}")
            self.action_complete_event.set()
            return False

        # Create joint angle list
        joint_angles = []
        for i, angle in enumerate(joint_positions):
            joint_angles.append({"joint_identifier": i, "value": float(angle)})

        # Create the proper message structure
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

            # Wait for completion if requested
            if wait_for_completion:
                print("Waiting for trajectory to complete...")
                if self.wait_for_action_completion():
                    print("Trajectory completed")
                    return True
                else:
                    print("Timed out waiting for trajectory completion")
                    return False
            return True
        except Exception as e:
            print(f"Failed to send joint trajectory command: {str(e)}")
            self.action_complete_event.set()  # Reset event since action failed to start
            return False

    def send_cartesian_trajectory(
        self, cartesian_pose, wait_for_completion=True, speed_factor=10.0
    ):
        """
        Send a cartesian trajectory command using the play_cartesian_trajectory service
        cartesian_pose: List of 6 cartesian pose values [x, y, z, theta_x, theta_y, theta_z]
        wait_for_completion: If True, wait for the action to complete before returning
        speed_factor: Factor to increase the speed (1.0 is default speed, higher values increase speed)
        """
        # Wait for any previous action to complete
        if not self.action_complete_event.is_set():
            print("Waiting for previous action to complete...")
            if not self.wait_for_action_completion():
                print("Timed out waiting for previous action")
                return False

        # Reset completion event
        self.action_complete_event.clear()

        # Start time for profiling
        self.movement_start_time = time.time()

        # Convert to orientation component to degrees
        cartesian_pose[3:] = np.rad2deg(cartesian_pose[3:])
        # Offset the roll by 180 degrees
        cartesian_pose[3:] = np.rad2deg([-3.14, 0, 1.571])
        print(f"Sending cartesian trajectory: {cartesian_pose}")

        # Set speed constraint - based on the Kortex API structure
        # CartesianTrajectoryConstraint has an oneof_type field that can contain CartesianSpeed
        # CartesianSpeed has translation and orientation fields
        translation_speed = 0.3 * speed_factor  # Default translation speed in m/s
        orientation_speed = 20.0 * speed_factor  # Default orientation speed in deg/s

        # Create the proper message structure
        request = {
            "input": {
                "target_pose": {
                    "x": float(cartesian_pose[0]),
                    "y": float(cartesian_pose[1]),
                    "z": float(cartesian_pose[2]),
                    "theta_x": float(cartesian_pose[3]),
                    "theta_y": float(cartesian_pose[4]),
                    "theta_z": float(cartesian_pose[5]),
                },
                "constraint": {
                    "oneof_type": {
                        "speed": [
                            {
                                "translation": float(translation_speed),
                                "orientation": float(orientation_speed),
                            }
                        ]
                    }
                },
            }
        }

        try:
            _ = self.play_cartesian_trajectory_service.call(request)
            print(
                f"Cartesian trajectory command sent successfully with speed factor: {speed_factor}"
            )

            # Wait for completion if requested
            if wait_for_completion:
                print("Waiting for trajectory to complete...")
                if self.wait_for_action_completion():
                    movement_time = time.time() - self.movement_start_time
                    print(f"Trajectory completed in {movement_time:.3f} seconds")
                    return True
                else:
                    print("Timed out waiting for trajectory completion")
                    return False
            return True
        except Exception as e:
            print(f"Failed to send cartesian trajectory command: {str(e)}")
            self.action_complete_event.set()  # Reset event since action failed to start
            return False

    def get_last_movement_duration(self):
        """Get the duration of the last completed movement in seconds"""
        return self.last_movement_duration

    def close(self):
        """Close all connections to the robot"""
        if hasattr(self, "joint_state_sub") and self.joint_state_sub:
            self.joint_state_sub.unsubscribe()

        if hasattr(self, "base_feedback_sub") and self.base_feedback_sub:
            self.base_feedback_sub.unsubscribe()

        if hasattr(self, "action_notif_sub") and self.action_notif_sub:
            self.action_notif_sub.unsubscribe()

        if hasattr(self, "client") and self.client.is_connected:
            self.client.terminate()

        print("ROS connections closed")
