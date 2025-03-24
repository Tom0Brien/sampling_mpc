import argparse

import evosax
import mujoco
import numpy as np
import jax.numpy as jnp

from hydrax.algs import MPPI, Evosax, PredictiveSampling
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

    # Set the controller based on command-line arguments
    if args.algorithm == "ps" or args.algorithm is None:
        print("Running predictive sampling")
        ctrl = PredictiveSampling(
            task,
            num_samples=1,
            noise_level=0.01,
        )

    elif args.algorithm == "mppi":
        print("Running MPPI")
        if control_mode == ControlMode.GENERAL_VI:
            # Individual gain optimization (6 controls + 12 gains)
            ctrl = MPPI(
                task,
                num_samples=2000,
                noise_level=np.array(
                    [
                        0.05,  # x reference noise level
                        0.05,  # y reference noise level
                        0.05,  # z reference noise level
                        0.01,  # roll reference noise level
                        0.01,  # pitch reference noise level
                        0.01,  # yaw reference noise level
                        1,  # kp x noise level
                        1,  # kp y noise level
                        1,  # kp z noise level
                        1,  # kp roll noise level
                        1,  # kp pitch noise level
                        1,  # kp yaw noise level
                        1,  # kd x noise level
                        1,  # kd y noise level
                        1,  # kd z noise level
                        1,  # kd roll noise level
                        1,  # kd pitch noise level
                        1,  # kd yaw noise level
                    ]
                ),
                temperature=0.001,
            )
        elif control_mode == ControlMode.CARTESIAN_SIMPLE_VI:
            # Simple gain optimization (6 controls + 2 gains)
            ctrl = MPPI(
                task,
                num_samples=2000,
                noise_level=np.array(
                    [
                        0.05,  # x reference noise level
                        0.05,  # y reference noise level
                        0.05,  # z reference noise level
                        0.01,  # roll reference noise level
                        0.01,  # pitch reference noise level
                        0.01,  # yaw reference noise level
                        1,  # translational p-gain noise level
                        1,  # rotational p-gain noise level
                    ]
                ),
                temperature=0.001,
            )
        else:
            ctrl = MPPI(
                task,
                num_samples=1000,
                noise_level=np.array(
                    [
                        0.05,  # x reference noise level
                        0.05,  # y reference noise level
                        0.05,  # z reference noise level
                        0.01,  # roll reference noise level
                        0.01,  # pitch reference noise level
                        0.01,  # yaw reference noise level
                    ]
                ),
                temperature=0.001,
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

    # Define the initial control (tile to planning horizon) if needed
    initial_control = None
    if control_mode == ControlMode.CARTESIAN or control_mode == ControlMode.GENERAL:
        initial_control = jnp.tile(
            jnp.array(
                [
                    0.5,  # x reference
                    0.0,  # y reference
                    0.4,  # z reference
                    -3.14,  # roll reference
                    0.0,  # pitch reference
                    0.0,  # yaw reference
                ]
            ),
            (task.planning_horizon, 1),
        )
    elif control_mode == ControlMode.CARTESIAN_SIMPLE_VI:
        initial_control = jnp.tile(
            jnp.array([0.5, 0.0, 0.4, -3.14, 0.0, 0.0, 300, 50]),
            (task.planning_horizon, 1),
        )

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
