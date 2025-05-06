from franka import FrankaRosInterface
import time


def main():
    franka = FrankaRosInterface()
    while franka.client.is_connected:
        franka.send_cartesian_command([0.5, 0.0, 0.4])
        time.sleep(1)


if __name__ == "__main__":
    print("Starting move to start")
    main()
