from kinova import KinovaRosInterface
import time


def main():
    # Create the interface with the robot's namespace
    kinova = KinovaRosInterface(robot_name="my_gen3")

    try:
        # Move to a starting position
        print("Moving to starting position...")
        kinova.send_cartesian_trajectory(
            [4.5816183e-01, -3.2090186e-03, 4.9384654e-01, -3.14, 0, 1.571]
        )
        for i in range(1000):
            kinova.send_cartesian_trajectory(
                [4.5816183e-01, -3.2090186e-03, 4.9384654e-01, -3.14, 0, 1.571]
            )

        # kinova.send_joint_trajectory(
        #     [-0.00877, 0.368, 3.15, -1.45, -0.00451, -1.33, 1.58]
        # )
        time.sleep(1)

        print("Move to start completed successfully")
    except Exception as e:
        print(f"Error during movement: {e}")
    finally:
        # Always close the interface properly
        kinova.close()


if __name__ == "__main__":
    print("Starting move to start for Kinova Gen3")
    main()
