# sampling_mpc

Exploring Sampling based MPC. Building upon [`hydrax`](https://github.com/vincekurtz/hydrax)

# Setup uv environment

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), a faster alternative to `pip`
2. Create a virtual environment: `uv venv --python 3.11`
3. Activate it: `source .venv/bin/activate`
4. Install `uv pip install -e ".[all]"`

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

Now we have access to the full Franka ROS interface documented [`here`](https://frankaemika.github.io/docs/franka_ros.html).
