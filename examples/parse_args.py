import argparse


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of the push-T task."
    )
    parser.add_argument(
        "--gain-mode",
        type=str,
        choices=["none", "individual", "simple"],
        default="none",
        help="Gain optimization mode (none, individual, or simple)",
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
