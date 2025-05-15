from kinova import KinovaRosInterface
import time


def main():
    # Create the interface with the robot's namespace
    kinova = KinovaRosInterface(robot_name="my_gen3")

    try:
        # Move to a starting position
        print("Moving to starting position...")
        kinova.send_cartesian_trajectory(
            [0.45666519, 0.0013501, 0.43372431, 1.571, 0, 1.571]
        )
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
