#!/usr/bin/env python3
import roslibpy
import time
import numpy as np
from scipy.spatial.transform import Rotation


class FrankaRosInterface:
    """
    Interface for controlling Franka robot using ROS for communication.
    This class handles the ROS connection, state tracking, and sending commands.
    """

    def __init__(self, host="localhost", port=9090):
        # ROS connection
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        print(f"Connected to ROS: {self.client.is_connected}")

        # Setup subscribers for robot state
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

        # Setup gripper publishers
        self.gripper_move_pub = roslibpy.Topic(
            self.client, "/franka_gripper/move/goal", "franka_gripper/MoveActionGoal"
        )
        self.gripper_grasp_pub = roslibpy.Topic(
            self.client, "/franka_gripper/grasp/goal", "franka_gripper/GraspActionGoal"
        )
        self.gripper_move_pub.advertise()
        self.gripper_grasp_pub.advertise()

        # Setup dynamic reconfigure compliance service
        self.compliance_param_client = roslibpy.Service(
            self.client,
            "/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node/set_parameters",
            "dynamic_reconfigure/Reconfigure",
        )

        # Wait for publishers to be ready
        time.sleep(1.0)

        # Latest state information
        self.q = None
        self.dq = None
        self.ee_position = np.zeros(3)
        self.ee_velocity = np.zeros(6)

        # Initial compliance parameters
        self.translational_stiffness = 200.0  # N/m
        self.rotational_stiffness = 10.0  # Nm/rad
        self.nullspace_stiffness = 10.0

        # Set up franka state callback
        self.franka_state_sub.subscribe(self.franka_state_callback)

        # Wait for first joint state message
        print("Waiting for joint state messages...")
        while self.q is None:
            time.sleep(0.1)
        print("Received initial joint state")

        # Send initial compliance parameters
        self.update_compliance_params(
            self.translational_stiffness,
            self.rotational_stiffness,
            self.nullspace_stiffness,
        )

        # Close the gripper during initialization
        print("Closing the gripper...")
        self.close_gripper()

    def franka_state_callback(self, message):
        """Callback for franka state messages"""
        # Extract end-effector position from franka state message
        # This is a simplified version - in practice you might want to
        # use forward kinematics or extract from the tf tree
        self.q = np.array(message["q"])
        self.dq = np.array(message["dq"])

        # If you have O_T_EE (end effector transform), you can extract position:
        if "O_T_EE" in message:
            O_T_EE = np.transpose(np.reshape(message["O_T_EE"], (4, 4)))
            self.ee_position = O_T_EE[:3, 3]  # Translation part

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
        else:
            print("Failed to update compliance parameters:", result)

    def send_cartesian_command(self, position, orientation=None):
        """
        Send a cartesian pose command to the robot
        position: [x, y, z]
        orientation: [r, p, y] or None for default orientation
        """
        # Use default orientation if none provided
        if orientation is None:
            quat = Rotation.from_euler("xyz", [-3.14, 0.0, 0.0]).as_quat()
        else:
            quat = Rotation.from_euler("xyz", orientation).as_quat()

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
                    "x": float(
                        position[0] + 0.013
                    ),  # OFFSET - TODO: Figure out why this is needed
                    "y": float(position[1]),
                    "z": float(
                        position[2] - 0.007
                    ),  # OFFSET - TODO: Figure out why this is needed
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

    def close_gripper(self, width=0.03, epsilon=0.005, speed=0.1, force=5.0):
        """Close the gripper with the specified parameters"""
        grasp_msg = {
            "goal": {
                "width": width,
                "epsilon": {"inner": epsilon, "outer": epsilon},
                "speed": speed,
                "force": force,
            }
        }
        self.gripper_grasp_pub.publish(roslibpy.Message(grasp_msg))
        # Allow some time for the gripper to close
        time.sleep(3.0)
        print("Gripper close command sent")

    def open_gripper(self, width=0.08, speed=0.1):
        """Open the gripper with the specified parameters"""
        move_msg = {"goal": {"width": width, "speed": speed}}
        self.gripper_move_pub.publish(roslibpy.Message(move_msg))
        # Allow some time for the gripper to open
        time.sleep(3.0)
        print("Gripper open command sent")

    def close(self):
        """Close all connections"""
        self.cartesian_pose_pub.unadvertise()
        self.franka_state_sub.unsubscribe()
        self.gripper_move_pub.unadvertise()
        self.gripper_grasp_pub.unadvertise()
        self.client.terminate()
        print("ROS connections closed")
