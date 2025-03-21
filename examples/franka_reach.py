import argparse

import evosax
import mujoco
import numpy as np

from hydrax.algs import MPPI, Evosax, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.franka_reach import FrankaReach
from hydrax.task_base import GainOptimizationMode
from parse_args import parse_args

"""
Run an interactive simulation of the Franka reaching task.

Double click on the green target, then drag it around with [ctrl + right-click].
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
    task = FrankaReach(gain_mode=gain_mode)

    # Print control dimensions
    if gain_mode == GainOptimizationMode.INDIVIDUAL:
        print(
            f"Control dimensions: {task.nu_ctrl} (controls) + {2 * task.nu_ctrl} (gains) = {task.nu_total}"
        )
    elif gain_mode == GainOptimizationMode.SIMPLE:
        print(
            f"Control dimensions: {task.nu_ctrl} (controls) + 2 (trans/rot p-gains) = {task.nu_total}"
        )
    else:
        print(f"Control dimensions: {task.model.nu}")

    # Set the controller based on command-line arguments
    if args.algorithm == "ps" or args.algorithm is None:
        print("Running predictive sampling")
        ctrl = PredictiveSampling(
            task,
            num_samples=128,
            noise_level=0.05,
        )

    elif args.algorithm == "mppi":
        print("Running MPPI")
        if gain_mode == GainOptimizationMode.INDIVIDUAL:
            # Individual gain optimization (6 controls + 12 gains)
            ctrl = MPPI(
                task,
                num_samples=2000,
                noise_level=np.array(
                    [
                        0.01,  # x reference noise level
                        0.01,  # y reference noise level
                        0.01,  # z reference noise level
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
        elif gain_mode == GainOptimizationMode.SIMPLE:
            # Simple gain optimization (6 controls + 2 gains)
            ctrl = MPPI(
                task,
                num_samples=2000,
                noise_level=np.array(
                    [
                        0.01,  # x reference noise level
                        0.01,  # y reference noise level
                        0.01,  # z reference noise level
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
            ctrl = MPPI(task, num_samples=2000, noise_level=0.01, temperature=0.001)

    elif args.algorithm == "cmaes":
        print("Running CMA-ES")
        ctrl = Evosax(task, evosax.Sep_CMA_ES, num_samples=128, elite_ratio=0.5)

    elif args.algorithm == "samr":
        print("Running genetic algorithm with Self-Adaptation Mutation Rate (SAMR)")
        ctrl = Evosax(task, evosax.SAMR_GA, num_samples=128)

    elif args.algorithm == "de":
        print("Running Differential Evolution (DE)")
        ctrl = Evosax(task, evosax.DE, num_samples=128)

    elif args.algorithm == "gld":
        print("Running Gradient-Less Descent (GLD)")
        ctrl = Evosax(task, evosax.GLD, num_samples=128)

    elif args.algorithm == "diffusion":
        print("Running Diffusion Evolution")
        es_params = evosax.strategies.diffusion.EvoParams(
            sigma_init=0.01, scale_factor=0.1, fitness_map_temp=1.0
        )
        ctrl = Evosax(
            task, evosax.DiffusionEvolution, num_samples=128, es_params=es_params
        )

    elif args.algorithm == "rs":
        print("Running uniform random search")
        es_params = evosax.strategies.random.EvoParams(
            range_min=-1.0,
            range_max=1.0,
        )
        ctrl = Evosax(task, evosax.RandomSearch, num_samples=128, es_params=es_params)
    else:
        parser = argparse.ArgumentParser()
        parser.error("Invalid algorithm")

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    # Set the initial joint positions
    mj_data.qpos[:7] = [-0.196, -0.189, 0.182, -2.1, 0.0378, 1.91, 0.756]

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=10,
        show_traces=True,
        max_traces=5,
        show_debug_info=True,
        record_video=True,
        plot_costs=True,
    )


if __name__ == "__main__":
    # This is the critical line that prevents the multiprocessing error
    from multiprocessing import freeze_support

    freeze_support()
    main()
