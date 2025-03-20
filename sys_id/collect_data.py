#!/usr/bin/env python3
import roslibpy
import numpy as np
import time
import os
from datetime import datetime
from scipy.spatial.transform import Rotation
import argparse


class RosDataCollector:
    def __init__(self, host="localhost", port=9090):
        # Connect to ROS
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        print(f"Connected to ROS: {self.client.is_connected}")

        # Set up subscribers for robot state
        self.joint_state_sub = roslibpy.Topic(
            self.client, "/joint_states", "sensor_msgs/JointState"
        )
        self.cartesian_pose_sub = roslibpy.Topic(
            self.client,
            "/cartesian_impedance_example_controller/equilibrium_pose",
            "geometry_msgs/PoseStamped",
        )

        # Add subscriber for measured end-effector pose from Franka state
        self.franka_state_sub = roslibpy.Topic(
            self.client,
            "/franka_state_controller/franka_states",
            "franka_msgs/FrankaState",
        )

        # Data storage
        self.joint_positions = []
        self.joint_velocities = []
        self.commanded_ee_poses = []  # Renamed from cartesian_poses for clarity
        self.measured_ee_poses = []  # New array for measured end-effector poses
        self.timestamps = []
        self.controller_params = {}

        # Subscribe to topics
        self.joint_state_sub.subscribe(self.joint_state_callback)
        self.cartesian_pose_sub.subscribe(self.cartesian_pose_callback)
        self.franka_state_sub.subscribe(self.franka_state_callback)

        # Setup publisher for cartesian pose commands
        self.cartesian_pose_pub = roslibpy.Topic(
            self.client,
            "/cartesian_impedance_example_controller/equilibrium_pose",
            "geometry_msgs/PoseStamped",
            queue_size=10,
        )
        self.cartesian_pose_pub.advertise()

        # Setup service client for dynamic reconfigure to set compliance parameters
        self.compliance_param_client = roslibpy.Service(
            self.client,
            "/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node/set_parameters",
            "dynamic_reconfigure/Reconfigure",
        )

    def set_compliance_params(
        self, translational_stiffness, rotational_stiffness, nullspace_stiffness=10.0
    ):
        """
        Set the compliance parameters of the cartesian impedance controller
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
            f"Setting compliance parameters: trans={translational_stiffness}, rot={rotational_stiffness}, null={nullspace_stiffness}"
        )

        # Store the parameters
        self.controller_params = {
            "translational_stiffness": translational_stiffness,
            "rotational_stiffness": rotational_stiffness,
            "nullspace_stiffness": nullspace_stiffness,
        }

        # Call the service
        self.compliance_param_client.call(request, self.compliance_service_callback)

    def compliance_service_callback(self, result):
        """Callback for the compliance parameter service call"""
        if "config" in result:
            print("Successfully updated compliance parameters")
        else:
            print("Failed to update compliance parameters:", result)

    def joint_state_callback(self, message):
        self.joint_positions.append(np.array(message["position"]))
        self.joint_velocities.append(np.array(message["velocity"]))
        if "header" in message:
            self.timestamps.append(
                message["header"]["stamp"]["secs"]
                + message["header"]["stamp"]["nsecs"] * 1e-9
            )

    def cartesian_pose_callback(self, message):
        """Callback for the commanded equilibrium pose"""
        pose = message["pose"]
        position = [pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]]
        orientation = [
            pose["orientation"]["x"],
            pose["orientation"]["y"],
            pose["orientation"]["z"],
            pose["orientation"]["w"],
        ]
        self.commanded_ee_poses.append(np.concatenate([position, orientation]))

    def franka_state_callback(self, message):
        """Callback for the measured end-effector state from Franka"""
        # Extract position from O_T_EE transformation matrix
        if "O_T_EE" in message:
            # The transformation matrix is stored as a flattened 16-element array
            # For some reason, the transformation matrix is transposed in the message
            transform_matrix = np.transpose(np.reshape(message["O_T_EE"], (4, 4)))

            # Extract position from the transformation matrix
            position = transform_matrix[:3, 3]

            # Extract rotation matrix and convert to quaternion
            rotation_matrix = transform_matrix[:3, :3]
            orientation = Rotation.from_matrix(rotation_matrix).as_quat()

            # Combine into a single array
            ee_pose = np.concatenate([position, orientation])
            self.measured_ee_poses.append(ee_pose)
        else:
            print("Warning: No 'O_T_EE' field in franka_states message")

    def send_cartesian_command(self, position, orientation):
        """
        Send a cartesian pose command to the robot
        position: [x, y, z]
        orientation: [x, y, z, w] quaternion
        """

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
                    "x": float(orientation[0]),
                    "y": float(orientation[1]),
                    "z": float(orientation[2]),
                    "w": float(orientation[3]),
                },
            },
        }

        # Publish the pose message
        self.cartesian_pose_pub.publish(roslibpy.Message(pose_msg))

    def collect_data(self, duration=10.0):
        """Collect data for specified duration in seconds"""
        print(f"Collecting data for {duration} seconds...")
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.01)  # Small sleep to prevent CPU hogging

        # Convert lists to numpy arrays
        joint_positions = np.array(self.joint_positions)
        joint_velocities = np.array(self.joint_velocities)
        commanded_ee_poses = np.array(self.commanded_ee_poses)
        measured_ee_poses = (
            np.array(self.measured_ee_poses) if self.measured_ee_poses else None
        )

        print(f"Collected {len(joint_positions)} joint state data points")
        print(f"Collected {len(commanded_ee_poses)} commanded poses")
        print(
            f"Collected {len(measured_ee_poses) if measured_ee_poses is not None else 0} measured end-effector poses"
        )

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "commanded_ee_poses": commanded_ee_poses,
            "measured_ee_poses": measured_ee_poses,
            "timestamps": np.array(self.timestamps) if self.timestamps else None,
            "controller_params": self.controller_params,
        }

    def cleanup(self):
        self.cartesian_pose_pub.unadvertise()
        self.joint_state_sub.unsubscribe()
        self.cartesian_pose_sub.unsubscribe()
        self.franka_state_sub.unsubscribe()
        self.client.terminate()


def generate_sinusoidal_trajectory(
    base_pose,
    duration,
    frequency=0.1,
    amplitude_pos=0.05,
    amplitude_ori=0.1,
    control_rate=10,
):
    """
    Generate a sinusoidal trajectory around a base pose

    Args:
        base_pose: Base pose [x, y, z, qx, qy, qz, qw]
        duration: Duration in seconds
        frequency: Frequency of oscillation in Hz
        amplitude_pos: Amplitude of position oscillation in meters
        amplitude_ori: Amplitude of orientation oscillation in radians
        control_rate: Rate of control commands in Hz

    Returns:
        times: Array of timestamps
        poses: Array of poses [x, y, z, qx, qy, qz, qw]
    """
    # Number of points in the trajectory
    num_points = int(duration * control_rate)

    # Time vector
    times = np.linspace(0, duration, num_points)

    # Extract base position and orientation
    base_pos = base_pose[:3]
    base_quat = base_pose[3:7]
    base_rot = Rotation.from_quat(base_quat)
    base_euler = base_rot.as_euler("xyz")

    # Initialize pose array
    poses = np.zeros((num_points, 7))

    # Generate trajectory
    for i, t in enumerate(times):
        # Sinusoidal position offsets
        pos_offset = np.array(
            [
                amplitude_pos * np.sin(2 * np.pi * frequency * t),
                amplitude_pos
                * np.sin(2 * np.pi * frequency * t + np.pi / 3),  # Phase shift
                amplitude_pos
                * np.sin(2 * np.pi * frequency * t + 2 * np.pi / 3),  # Different phase
            ]
        )

        # Sinusoidal orientation offsets (in Euler angles)
        ori_offset = np.array(
            [
                amplitude_ori * np.sin(2 * np.pi * frequency * t),
                amplitude_ori * np.sin(2 * np.pi * frequency * t + np.pi / 3),
                amplitude_ori * np.sin(2 * np.pi * frequency * t + 2 * np.pi / 3),
            ]
        )

        # Compute new position
        new_pos = base_pos + pos_offset

        # Compute new orientation
        new_euler = base_euler + ori_offset
        new_quat = Rotation.from_euler("xyz", new_euler).as_quat()

        # Store pose
        poses[i, :3] = new_pos
        poses[i, 3:7] = new_quat

    return times, poses


def main():
    parser = argparse.ArgumentParser(
        description="Collect Franka robot data with sinusoidal movements"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="ROS websocket host"
    )
    parser.add_argument("--port", type=int, default=9090, help="ROS websocket port")
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration to collect data (seconds)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=0.1,
        help="Frequency of sinusoidal movement (Hz)",
    )
    parser.add_argument(
        "--amplitude-pos",
        type=float,
        default=0.05,
        help="Amplitude of position movement (m)",
    )
    parser.add_argument(
        "--amplitude-ori",
        type=float,
        default=0.1,
        help="Amplitude of orientation movement (rad)",
    )
    parser.add_argument(
        "--control-rate", type=float, default=10.0, help="Control command rate (Hz)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sys_id/collected_data",
        help="Directory to save data",
    )
    parser.add_argument(
        "--trans-stiffness",
        type=float,
        default=200.0,
        help="Translational stiffness for the cartesian impedance controller",
    )
    parser.add_argument(
        "--rot-stiffness",
        type=float,
        default=10.0,
        help="Rotational stiffness for the cartesian impedance controller",
    )
    parser.add_argument(
        "--null-stiffness",
        type=float,
        default=1.0,
        help="Nullspace stiffness for the cartesian impedance controller",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Connect to ROS
    collector = RosDataCollector(host=args.host, port=args.port)
    print("Connected to ROS, waiting for initial state...")

    # Wait a bit for initial robot state
    time.sleep(1.0)

    # Set the compliance parameters
    collector.set_compliance_params(
        args.trans_stiffness, args.rot_stiffness, args.null_stiffness
    )
    print("Waiting for compliance parameters to take effect...")
    time.sleep(1.0)

    # Check if we have received any commanded ee poses
    if len(collector.commanded_ee_poses) == 0:
        print("WARNING: No commanded poses received. Using default pose.")
        base_pose = np.array([0.5, 0.0, 0.5, -1.0, 0.0, 0.0, 0.0])  # Default pose
    else:
        base_pose = collector.commanded_ee_poses[-1]
        print(f"Using current robot pose as base: {base_pose}")

    # Clear any cached data before we start the experiment
    collector.joint_positions = []
    collector.joint_velocities = []
    collector.commanded_ee_poses = []
    collector.measured_ee_poses = []
    collector.timestamps = []
    # Generate sinusoidal trajectory
    print(
        f"Generating sinusoidal trajectory for {args.duration}s with frequency {args.frequency}Hz..."
    )
    times, poses = generate_sinusoidal_trajectory(
        base_pose,
        args.duration,
        frequency=args.frequency,
        amplitude_pos=args.amplitude_pos,
        amplitude_ori=args.amplitude_ori,
        control_rate=args.control_rate,
    )

    # Execute trajectory and collect data
    print(f"Executing trajectory and collecting data for {args.duration}s...")
    start_time = time.time()
    period = 1.0 / args.control_rate

    for i, pose in enumerate(poses):
        # Calculate elapsed time
        elapsed = time.time() - start_time

        # If we're ahead of schedule, sleep
        if elapsed < times[i]:
            time.sleep(times[i] - elapsed)

        # Send command
        collector.send_cartesian_command(pose[:3], pose[3:7])

        # Print progress
        if i % 10 == 0:
            progress = 100 * i / len(poses)
            print(f"Progress: {progress:.1f}%", end="\r")

    # Wait for final commands to be executed
    print("\nFinishing execution, waiting 2 seconds...")
    time.sleep(2.0)

    # Collect data
    data = collector.collect_data(
        duration=0.0
    )  # Just collect what we have, don't wait more

    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"franka_sinusoidal_data_{timestamp}.npy"
    filepath = os.path.join(args.output_dir, filename)
    np.save(filepath, data)
    print(f"Saved data to {filepath}")

    # Print a summary of the collected data
    print("\nData Summary:")
    print(f"  Controller Parameters:")
    for key, value in data["controller_params"].items():
        print(f"    {key}: {value}")
    print(f"  Recorded joint states: {len(data['joint_positions'])}")
    print(f"  Recorded commanded poses: {len(data['commanded_ee_poses'])}")
    print(
        f"  Recorded measured poses: {len(data['measured_ee_poses']) if data['measured_ee_poses'] is not None else 0}"
    )

    # Clean up
    collector.cleanup()
    print("Data collection complete!")


if __name__ == "__main__":
    main()
