# sampling_mpc

Exploring Sampling based MPC. Building upon [`hydrax`](https://github.com/vincekurtz/hydrax)

# Setup uv environment

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), a faster alternative to `pip`
2. Initialize and update git submodules: `git submodule update --init --recursive`
3. Create a virtual environment: `uv venv --python 3.11`
4. Activate it: `source .venv/bin/activate`
5. Install project dependencies: `uv pip install -e ".[all]"`
6. Make sure hydrax is properly installed: `uv pip install -e ./hydrax-core`

# Dockerized Franka ROS interface

This repository allows you to run [`libfranka`](https://frankaemika.github.io/docs/libfranks.html) and [`franka_ros`](https://frankaemika.github.io/docs/franka_ros.html) in a Docker environment (complete with ROS Noetic, GUI access, and networking to the robot with the `FCI_IP` environment variable).

To build:

```bash
docker compose build
```

To run

```bash
docker compose up
```

To enter shell inside container

```bash
docker compose main /bin/bash
```

**_Note: For GUI applications (RVIS, Gazebo, etc...) to work, you may need to run `xhost +local:` to allow the Docker container to access the X server on your host machine._**

Now we have access to the full Franka ROS interface documented [`here`](https://frankaemika.github.io/docs/franka_ros.html).
