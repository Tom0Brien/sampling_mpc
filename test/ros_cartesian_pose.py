#!/usr/bin/env python3
import roslibpy
import time
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    # Connect to ROS
    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()

    print("Is ROS connected?", client.is_connected)

    # Create publisher for cartesian pose commands
    cartesian_pose_pub = roslibpy.Topic(
        client,
        "/cartesian_impedance_example_controller/equilibrium_pose",
        "geometry_msgs/PoseStamped",
        queue_size=10,
    )

    # Advertise the publisher
    cartesian_pose_pub.advertise()

    # Wait for publisher to be ready
    time.sleep(1.0)

    # Create a simple pose message
    # Position in meters, orientation as quaternion [x, y, z, w]
    position = [0.5, 0.0, 0.5]  # x, y, z in meters

    # Create quaternion from Euler angles (roll, pitch, yaw)
    euler_angles = [-3.14, 0.0, 0.0]  # roll, pitch, yaw in radians
    rotation = Rotation.from_euler("xyz", euler_angles)
    quaternion = rotation.as_quat()  # Returns [x, y, z, w]

    # Create the pose message
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
                "x": float(quaternion[0]),
                "y": float(quaternion[1]),
                "z": float(quaternion[2]),
                "w": float(quaternion[3]),
            },
        },
    }

    # Print what we're sending
    print(f"Sending cartesian pose command:")
    print(f"Position: {position}")
    print(f"Orientation (quaternion): {quaternion}")

    # Publish the pose message
    cartesian_pose_pub.publish(roslibpy.Message(pose_msg))

    # Wait a bit to ensure the message is sent
    time.sleep(1.0)

    # Cleanup
    cartesian_pose_pub.unadvertise()
    print("Cartesian pose command sent")

    client.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down")
