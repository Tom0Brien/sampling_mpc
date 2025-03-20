import argparse

import evosax
import mujoco
import numpy as np

from hydrax.algs import MPPI, Evosax, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.particle_box import ParticleBox
from hydrax.task_base import GainOptimizationMode
from parse_args import parse_args

"""
Run an interactive simulation of the particle with box constraint task.

Double click on the green target, then drag it around with [ctrl + right-click].
The particle (red) should reach the target while keeping the box (orange) within the blue safe area.
"""


def main():
    args = parse_args()

    # Map the gain mode string to the enum
    gain_mode_map = {
        "none": GainOptimizationMode.NONE,
        "individual": GainOptimizationMode.INDIVIDUAL,
        "simple": GainOptimizationMode.SIMPLE,
    }
    gain_mode = gain_mode_map[args.gain_mode]

    # Define the task (cost and dynamics)
    task = ParticleBox(gain_mode=gain_mode)

    # Print control dimensions
    if gain_mode == GainOptimizationMode.INDIVIDUAL:
        print(
            f"Control dimensions: {task.nu_ctrl} (controls) + {2 * task.nu_ctrl} (gains) = {task.nu_total}"
        )
    elif gain_mode == GainOptimizationMode.SIMPLE:
        # Not supported for this task
        raise ValueError("Simple gain mode not supported for this task")
    else:
        print(f"Control dimensions: {task.model.nu}")

    # Set the controller based on command-line arguments
    if args.algorithm == "ps" or args.algorithm is None:
        print("Running predictive sampling")
        ctrl = PredictiveSampling(
            task,
            num_samples=2000,
            noise_level=0.001,
        )

    elif args.algorithm == "mppi":
        print("Running MPPI")
        if gain_mode == GainOptimizationMode.INDIVIDUAL:
            ctrl = MPPI(
                task,
                num_samples=200,
                noise_level=np.array([0.05, 0.05, 1, 1, 1, 1]),
                temperature=0.001,
            )
        elif gain_mode == GainOptimizationMode.SIMPLE:
            # Not supported for this task
            raise ValueError("Simple gain mode not supported for this task")
        else:
            ctrl = MPPI(task, num_samples=2000, noise_level=0.05, temperature=0.001)

    elif args.algorithm == "cmaes":
        print("Running CMA-ES")
        ctrl = Evosax(task, evosax.Sep_CMA_ES, num_samples=16, elite_ratio=0.5)

    elif args.algorithm == "samr":
        print("Running genetic algorithm with Self-Adaptation Mutation Rate (SAMR)")
        ctrl = Evosax(task, evosax.SAMR_GA, num_samples=16)

    elif args.algorithm == "de":
        print("Running Differential Evolution (DE)")
        ctrl = Evosax(task, evosax.DE, num_samples=16)

    elif args.algorithm == "gld":
        print("Running Gradient-Less Descent (GLD)")
        ctrl = Evosax(task, evosax.GLD, num_samples=16)

    elif args.algorithm == "diffusion":
        print("Running Diffusion")
        ctrl = Evosax(task, evosax.DiffusionEvolution, num_samples=1000)

    elif args.algorithm == "rs":
        print("Running uniform random search")
        es_params = evosax.strategies.random.EvoParams(
            range_min=-1.0,
            range_max=1.0,
        )
        ctrl = Evosax(task, evosax.RandomSearch, num_samples=16, es_params=es_params)
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
        show_traces=True,
        max_traces=5,
        record_video=True,
        plot_costs=True,
        show_debug_info=True,
    )


if __name__ == "__main__":
    # This is the critical line that prevents the multiprocessing error
    from multiprocessing import freeze_support

    freeze_support()
    main()
