import argparse

import evosax
import mujoco
import numpy as np

from mppii.algs import MPPI, Evosax, PredictiveSampling
from mppii.simulation.deterministic import run_interactive
from mppii.tasks.franka_reach import FrankaReach

"""
Run an interactive simulation of the particle tracking task.

Double click on the green target, then drag it around with [ctrl + right-click].
"""


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the particle tracking task."
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
subparsers.add_parser("diffusion", help="Diffusion Evolution")
args = parser.parse_args()

# Define the task (cost and dynamics)
task = FrankaReach(optimize_gains=args.optimize_gains)

# Print control dimensions
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
    if task.optimize_gains:
        ctrl = MPPI(
            task,
            num_samples=2000,
            noise_level=np.array(
                [
                    0.05,  # x reference noise level
                    0.05,  # y reference noise level
                    0.05,  # z reference noise level
                    0.05,  # roll reference noise level
                    0.05,  # pitch reference noise level
                    0.05,  # yaw reference noise level
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
    else:
        ctrl = MPPI(task, num_samples=2000, noise_level=0.05, temperature=0.001)

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
        sigma_init=0.01, scale_factor=0.1, fitness_map_temp=1.0
    )
    ctrl = Evosax(task, evosax.DiffusionEvolution, num_samples=128, es_params=es_params)
else:
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
)
