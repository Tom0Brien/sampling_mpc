import argparse

import evosax
import mujoco
import numpy as np
from hydrax.algs import MPPI, Evosax, PredictiveSampling, CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pushbox import PushBox
from hydrax.task_base import ControlMode
from parse_args import parse_args, control_mode_map

"""
Run an interactive simulation of the push-T task.

Double click on the green target to move the goal position.
"""


def main():
    args = parse_args()

    control_mode = control_mode_map[args.control_mode]

    # Define the task (cost and dynamics)
    task = PushBox(control_mode=control_mode)

    # Print control dimensions
    print(f"Control mode: {control_mode}")
    print(
        f"Control dimensions: {task.nu_ctrl} (controls) + {task.nu_total - task.nu_ctrl} (gains) = {task.nu_total}"
    )

    # Set the controller based on command-line arguments
    if args.algorithm == "ps" or args.algorithm is None:
        print("Running predictive sampling")
        ctrl = PredictiveSampling(
            task,
            num_samples=200,
            noise_level=0.001,
        )

    elif args.algorithm == "mppi":
        print("Running MPPI")
        if control_mode == ControlMode.GENERAL:
            ctrl = MPPI(
                task,
                num_samples=1000,
                noise_level=0.05,
                temperature=0.001,
            )
        elif control_mode == ControlMode.GENERAL_VI:
            ctrl = MPPI(
                task,
                num_samples=500,
                noise_level=np.array([0.2, 0.2, 0.3, 0.3, 0.3, 0.3]),
                temperature=0.001,
            )
        else:
            # Not supported for this task
            raise ValueError("Control mode not supported for this task")
    elif args.algorithm == "cem" or args.algorithm is None:
        print("Running CEM")
        ctrl = CEM(
            task, num_samples=2000, num_elites=20, sigma_start=0.1, sigma_min=0.1
        )

    elif args.algorithm == "cmaes":
        print("Running CMA-ES")
        ctrl = Evosax(task, evosax.Sep_CMA_ES, num_samples=1024, elite_ratio=0.5)

    elif args.algorithm == "samr":
        print("Running genetic algorithm with Self-Adaptation Mutation Rate (SAMR)")
        ctrl = Evosax(task, evosax.SAMR_GA, num_samples=1024)

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
            sigma_init=0.02, scale_factor=1
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
    mj_data.qpos = [0.0, -0.2, 0.04, 1.0, 0, 0, 0, 0.0, -0.3]

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=50,
        show_traces=False,
        max_traces=6,
        trace_width=6,
        trace_color=[0.0, 0.0, 1.0, 0.1],
        record_video=True,
        plot_costs=False,
        show_debug_info=False,
        keyboard_step_size=0.05,
    )


if __name__ == "__main__":
    # This is the critical line that prevents the multiprocessing error
    from multiprocessing import freeze_support

    freeze_support()
    main()
