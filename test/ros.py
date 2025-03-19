#!/usr/bin/env python3
import roslibpy
import time


def main():
    # Connect to ROS
    client = roslibpy.Ros(host="localhost", port=9090)
    client.run()

    print("Is ROS connected?", client.is_connected)

    # Create publishers for the gripper actions
    move_pub = roslibpy.Topic(
        client, "/franka_gripper/move/goal", "franka_gripper/MoveActionGoal"
    )
    grasp_pub = roslibpy.Topic(
        client, "/franka_gripper/grasp/goal", "franka_gripper/GraspActionGoal"
    )

    # Advertise the publishers
    move_pub.advertise()
    grasp_pub.advertise()

    # Wait for publishers to be ready
    time.sleep(1.0)

    # Number of cycles to run
    cycles = 5

    for i in range(cycles):
        # Open the gripper (equivalent to the first rostopic pub command)
        print("Opening gripper...")
        move_msg = {"goal": {"width": 0.08, "speed": 0.1}}
        move_pub.publish(roslibpy.Message(move_msg))

        # Wait for the action to complete
        time.sleep(3.0)

        # Close/grasp with the gripper (equivalent to the second rostopic pub command)
        print("Closing gripper...")
        grasp_msg = {
            "goal": {
                "width": 0.03,
                "epsilon": {"inner": 0.005, "outer": 0.005},
                "speed": 0.1,
                "force": 5.0,
            }
        }
        grasp_pub.publish(roslibpy.Message(grasp_msg))

        # Wait for the action to complete
        time.sleep(3.0)

    # Cleanup
    move_pub.unadvertise()
    grasp_pub.unadvertise()
    print("Gripper control completed")

    client.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down")
