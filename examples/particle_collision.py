import argparse
import evosax
import mujoco
import numpy as np
import jax.numpy as jnp

from hydrax.algs import MPPI, Evosax, PredictiveSampling, CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.particle_collision import ParticleCollision
from hydrax.task_base import ControlMode
from parse_args import parse_args, control_mode_map

"""
Run an interactive simulation of the 3D particle collision avoidance task.

The green sphere represents the obstacle that can be moved with [ctrl + right-click].
The red sphere represents the fixed target position.
Use [shift + right-click] to rotate the view.
"""


def main():
    args = parse_args()

    # Map the control mode string to the enum
    control_mode = control_mode_map[args.control_mode]

    # Define the task (cost and dynamics)
    # Set a fixed target position and obstacle radius
    task = ParticleCollision(control_mode=control_mode)

    # Print control dimensions
    print(f"Control mode: {control_mode}")
    print(
        f"Control dimensions: {task.nu_ctrl} (controls) + {task.nu_total - task.nu_ctrl} (gains) = {task.nu_total}"
    )

    noise_level = None
    initial_control = None
    if control_mode == ControlMode.GENERAL:
        noise_level = 0.01
        initial_control = jnp.broadcast_to(
            jnp.array([0.4, 0.0, 0.5]), (task.planning_horizon, 3)
        )
    elif control_mode == ControlMode.GENERAL_VI:
        noise_level = np.array([0.01, 0.01, 0.01, 1, 1, 1])
        initial_control = jnp.broadcast_to(
            jnp.array([0.4, 0.0, 0.5, 1, 1, 1]), (task.planning_horizon, 6)
        )

    # Set the controller based on command-line arguments
    if args.algorithm == "ps" or args.algorithm is None:
        print("Running predictive sampling")
        ctrl = PredictiveSampling(
            task,
            num_samples=2000,
            noise_level=noise_level,
        )

    elif args.algorithm == "cem":
        print("Running CEM")
        ctrl = CEM(
            task,
            num_samples=128,
            num_elites=20,
            sigma_start=0.05,
            sigma_min=0.01,
            explore_fraction=0.0,
        )

    elif args.algorithm == "mppi":
        print("Running MPPI")
        ctrl = MPPI(task, num_samples=2000, noise_level=noise_level, temperature=0.001)

    elif args.algorithm == "cmaes":
        print("Running CMA-ES")
        ctrl = Evosax(task, evosax.Sep_CMA_ES, num_samples=128, elite_ratio=0.5)

    else:
        parser = argparse.ArgumentParser()
        parser.error("Invalid algorithm")

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=50,
        show_traces=True,  # Enable traces to see the particle's path
        max_traces=10,
        record_video=False,
        plot_costs=True,  # Enable cost plotting to see optimization progress
        show_debug_info=True,
        initial_control=initial_control,
    )


if __name__ == "__main__":
    # This is the critical line that prevents the multiprocessing error
    from multiprocessing import freeze_support

    freeze_support()
    main()
