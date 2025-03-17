import argparse

import evosax
import mujoco
import numpy as np

from mppii.algs import MPPI, Evosax, PredictiveSampling
from mppii.simulation.deterministic import run_interactive
from mppii.tasks.particle_corridor import ParticleCorridor

"""
Run an interactive simulation of the particle pushing a box through a corridor.

Double click on the green target, then drag it around with [ctrl + right-click].
The particle (red) should reach the target and push the box (orange) through the corridor
without letting the box touch the walls.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the particle pushing a box through a corridor."
)
parser.add_argument(
    "--optimize-gains",
    action="store_true",
    help="Optimize actuator gains along with control inputs",
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cmaes", help="CMA-ES")
subparsers.add_parser(
    "samr", help="Genetic Algorithm with Self-Adaptation Mutation Rate (SAMR)"
)
subparsers.add_parser("de", help="Differential Evolution")
subparsers.add_parser("gld", help="Gradient-Less Descent")
subparsers.add_parser("rs", help="Uniform Random Search")
subparsers.add_parser("diffusion", help="Diffusion")
args = parser.parse_args()

# Define the task (cost and dynamics)
task = ParticleCorridor(optimize_gains=args.optimize_gains)

# Print control dimensions
if args.optimize_gains:
    print(
        f"Control dimensions: {task.nu_ctrl} (controls) + {2 * task.nu_ctrl} (gains) = {task.nu_total}"
    )
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
    if task.optimize_gains:
        ctrl = MPPI(
            task,
            num_samples=200,
            noise_level=np.array([0.01, 0.01, 1, 1, 1, 1]),
            temperature=0.001,
        )
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
