import argparse

import evosax
import mujoco
import numpy as np
import jax.numpy as jnp

from hydrax.algs import MPPI, Evosax, PredictiveSampling, CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.franka_push import FrankaPush
from hydrax.task_base import ControlMode
from parse_args import parse_args, control_mode_map

"""
Run an interactive simulation of the box pushing task.

Double click on the green target to move the goal position.
"""


def main():
    args = parse_args()

    # Map the control mode string to the enum
    control_mode = control_mode_map[args.control_mode]

    # Define the task (cost and dynamics)
    task = FrankaPush(control_mode=control_mode)

    print(f"Control mode: {control_mode}")
    print(
        f"Control dimensions: {task.nu_ctrl} (controls) + {task.nu_total - task.nu_ctrl} (gains) = {task.nu_total}"
    )

    noise_level = None
    initial_control = None
    if control_mode == ControlMode.GENERAL_VI:
        print("Running general VI")
        noise_level = np.array([0.01] * 6 + [1] * 12)
        initial_control = jnp.tile(
            jnp.array([0.5, 0.0, 0.4, -3.14, 0.0, 0.0, 300, 50]),
            (task.planning_horizon, 1),
        )
    elif control_mode == ControlMode.CARTESIAN_SIMPLE_VI:
        print("Running cartesian simple VI")
        noise_level = np.array([0.01] * 6 + [1] * 2)
        initial_control = jnp.tile(
            jnp.array([0.5, 0.0, 0.4, -3.14, 0.0, 0.0, 300, 50]),
            (task.planning_horizon, 1),
        )
    else:
        print("Running general")
        noise_level = np.array([0.01] * 6)
        initial_control = jnp.tile(
            jnp.array([0.5, 0.0, 0.4, -3.14, 0.0, 0.0]),
            (task.planning_horizon, 1),
        )

    # Set the controller based on command-line arguments
    if args.algorithm == "ps" or args.algorithm is None:
        print("Running predictive sampling")
        ctrl = PredictiveSampling(
            task,
            num_samples=1,
            noise_level=noise_level,
        )

    elif args.algorithm == "mppi":
        print("Running MPPI")
        # Individual gain optimization (6 controls + 12 gains)
        ctrl = MPPI(
            task,
            num_samples=2000,
            noise_level=noise_level,
            temperature=0.001,
        )

    elif args.algorithm == "cem":
        print("Running CEM")
        ctrl = CEM(
            task, num_samples=2000, num_elites=20, sigma_start=0.1, sigma_min=0.1
        )

    elif args.algorithm == "cmaes":
        print("Running CMA-ES")
        ctrl = Evosax(task, evosax.Sep_CMA_ES, num_samples=128, elite_ratio=0.5)

    elif args.algorithm == "samr":
        print("Running genetic algorithm with Self-Adaptation Mutation Rate (SAMR)")
        ctrl = Evosax(task, evosax.SAMR_GA, num_samples=1024)

    elif args.algorithm == "gesmr":
        print("Running GESMR-GA")
        ctrl = Evosax(task, evosax.GESMR_GA, num_samples=1024)

    elif args.algorithm == "de":
        print("Running Differential Evolution (DE)")
        ctrl = Evosax(task, evosax.DE, num_samples=128)

    elif args.algorithm == "gld":
        print("Running Gradient-Less Descent (GLD)")
        ctrl = Evosax(task, evosax.GLD, num_samples=128)

    elif args.algorithm == "sa":
        print("Running Simulated Annealing")
        ctrl = Evosax(task, evosax.SimAnneal, num_samples=128)

    elif args.algorithm == "rs":
        print("Running uniform random search")
        es_params = evosax.strategies.random.EvoParams(
            range_min=-1.0,
            range_max=1.0,
        )
        ctrl = Evosax(task, evosax.RandomSearch, num_samples=128, es_params=es_params)

    elif args.algorithm == "diffusion":
        print("Running Diffusion Evolution")
        es_params = evosax.strategies.diffusion.EvoParams(
            sigma_init=0.01, scale_factor=0.1, fitness_map_temp=3.0
        )
        ctrl = Evosax(
            task,
            evosax.DiffusionEvolution,
            num_samples=1024,
            es_params=es_params,
        )
    else:
        parser = argparse.ArgumentParser()
        parser.error("Invalid algorithm")

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Set the initial joint positions
    mj_data.qpos[:9] = [
        -0.196,
        -0.189,
        0.182,
        -2.1,
        0.0378,
        1.91,
        0.756,
        0,
        0,
    ]

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=10,
        show_traces=True,
        max_traces=6,
        trace_width=6,
        trace_color=[0.0, 0.0, 1.0, 0.5],
        record_video=True,
        show_debug_info=True,
        plot_costs=True,
        initial_control=initial_control,
    )


if __name__ == "__main__":
    # This is the critical line that prevents the multiprocessing error
    from multiprocessing import freeze_support

    freeze_support()
    main()
