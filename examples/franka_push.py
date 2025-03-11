import argparse

import evosax
import mujoco

from mppii.algs import MPPI, Evosax, PredictiveSampling
from mppii.simulation.deterministic import run_interactive
from mppii.tasks.franka_push import FrankaPush

"""
Run an interactive simulation of the box pushing task.

Double click on the green target to move the goal position.
"""

# Define the task (cost and dynamics)
task = FrankaPush()

# Print control dimensions
print(f"Control dimensions: {task.model.nu}")

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the particle tracking task."
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
subparsers.add_parser("sa", help="Simulated Annealing")
subparsers.add_parser("diffusion", help="Diffusion Evolution")

args = parser.parse_args()

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
    ctrl = MPPI(task, num_samples=1000, noise_level=0.1, temperature=0.05)

elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(task, evosax.Sep_CMA_ES, num_samples=128, elite_ratio=0.5)

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
        sigma_init=0.01, scale_factor=0.1, fitness_map_temp=3.0
    )
    ctrl = Evosax(
        task,
        evosax.DiffusionEvolution,
        num_samples=1024,
        es_params=es_params,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
# Set the initial joint positions
# mj_data.qpos[:7] = [-0.199, -0.149, 0.179, -1.72, 0.028, 1.57, 0.763]
mj_data.qpos[:7] = [-0.196, -0.189, 0.182, -2.1, 0.0378, 1.91, 0.756]
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
)
