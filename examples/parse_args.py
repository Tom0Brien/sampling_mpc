import argparse

from hydrax.task_base import ControlMode

control_mode_map = {
    "general": ControlMode.GENERAL,
    "cartesian": ControlMode.CARTESIAN,
    "general_variable_impedance": ControlMode.GENERAL_VARIABLE_IMPEDANCE,
    "cartesian_variable_impedance": ControlMode.CARTESIAN_VARIABLE_IMPEDANCE,
    "cartesian_simple_variable_impedance": ControlMode.CARTESIAN_SIMPLE_VARIABLE_IMPEDANCE,
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
            "cartesian",
            "general_variable_impedance",
            "cartesian_variable_impedance",
            "cartesian_simple_variable_impedance",
        ],
        default="none",
        help="Control mode (none, general, cartesian, general_variable_impedance, cartesian_variable_impedance, cartesian_simple_variable_impedance)",
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
    return parser.parse_args()
