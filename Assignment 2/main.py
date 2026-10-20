import argparse
import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.run_classic_control import run_experiment as run_classic
from scripts.run_robotic_deterministic import train_arm_agent
from scripts.run_robotic_stochastic import train_stochastic_arm
from agents.dyna_q import DynaQAgent
from agents.prioritized_sweeping import PrioritizedSweepingAgent


def run_part_a(n_values, episodes):
    print("=" * 60)
    print("RUNNING PART (a): Classic Control Environments")
    print("=" * 60)
    envs = ["CartPole-v1", "MountainCar-v0"]
    agents = ["Dyna-Q", "PrioritizedSweeping"]

    for env_name in envs:
        print(f"\n--- Environment: {env_name} ---")
        for agent_type in agents:
            for n in n_values:
                ep_conv, avg_steps = run_classic(
                    env_name, agent_type, n, num_episodes=episodes
                )
                print(
                    f"[{agent_type}] n = {n:2d} | "
                    f"Episodes to Converge: {ep_conv} | "
                    f"Avg Steps/Episode (Final): {avg_steps:.1f}"
                )


def run_part_b1(n_values, episodes):
    print("\n" + "=" * 60)
    print("RUNNING PART (b)(i): Deterministic Robotic Arm Domain")
    print("=" * 60)

    for n in n_values:
        ep_dq, steps_dq = train_arm_agent(DynaQAgent, n, num_episodes=episodes)
        ep_ps, steps_ps = train_arm_agent(PrioritizedSweepingAgent, n, num_episodes=episodes)
        print(
            f"n = {n:2d} | Dyna-Q: {ep_dq} eps, {steps_dq:.1f} steps/ep | "
            f"Prioritized Sweeping: {ep_ps} eps, {steps_ps:.1f} steps/ep"
        )


def run_part_b2(n_values, episodes):
    print("\n" + "=" * 60)
    print("RUNNING PART (b)(ii): Non-Deterministic Robotic Arm Domain")
    print("=" * 60)

    for n in n_values:
        ep, steps = train_stochastic_arm(n, num_episodes=episodes)
        print(
            f"n = {n:2d} | Stochastic Dyna-Q: Converged in {ep} episodes | "
            f"Avg Steps/Episode (Final): {steps:.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run Model-based RL Benchmarks for Assignment 2"
    )
    parser.add_argument(
        "--part",
        type=str,
        choices=["a", "b1", "b2", "all"],
        default="all",
        help="Specific assignment part to execute (a, b1, b2, or all)",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[0, 5, 10, 20, 50],
        help="List of planning steps n to evaluate (e.g., --n 0 5 10 20 50)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Number of training episodes per run",
    )

    args = parser.parse_args()

    if args.part in ["a", "all"]:
        run_part_a(args.n, args.episodes)
    if args.part in ["b1", "all"]:
        run_part_b1(args.n, args.episodes)
    if args.part in ["b2", "all"]:
        run_part_b2(args.n, args.episodes)


if __name__ == "__main__":
    main()