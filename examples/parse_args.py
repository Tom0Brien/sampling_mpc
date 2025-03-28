import argparse

from hydrax.task_base import ControlMode

control_mode_map = {
    "general": ControlMode.GENERAL,
    "general_vi": ControlMode.GENERAL_VI,
    "cartesian": ControlMode.CARTESIAN,
    "cartesian_vi": ControlMode.CARTESIAN_VI,
    "cartesian_simple_vi": ControlMode.CARTESIAN_SIMPLE_VI,
}


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of the task."
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        choices=[
            "general",
            "general_vi",
            "cartesian",
            "cartesian_vi",
            "cartesian_simple_vi",
        ],
        default="general",
        help="Control mode (general, general_vi, cartesian, cartesian_vi, cartesian_simple_vi)",
    )
    subparsers = parser.add_subparsers(
        dest="algorithm", help="Sampling algorithm (choose one)"
    )
    subparsers.add_parser("ps", help="Predictive Sampling")
    subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
    subparsers.add_parser("cem", help="Cross-Entropy Method")
    subparsers.add_parser("cmaes", help="CMA-ES")
    subparsers.add_parser(
        "samr", help="Genetic Algorithm with Self-Adaptation Mutation Rate (SAMR)"
    )
    subparsers.add_parser("de", help="Differential Evolution")
    subparsers.add_parser("gld", help="Gradient-Less Descent")
    subparsers.add_parser("rs", help="Uniform Random Search")
    subparsers.add_parser("sa", help="Simulated Annealing")
    subparsers.add_parser("diffusion", help="Diffusion Evolution")
    return parser.parse_args()
