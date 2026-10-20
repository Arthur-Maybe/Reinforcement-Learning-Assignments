import sys
import os
import numpy as np
import gymnasium as gym

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.discretizer import StateDiscretizer
from agents.dyna_q import DynaQAgent
from agents.prioritized_sweeping import PrioritizedSweepingAgent


def run_experiment(env_name, agent_type, n, num_episodes=500):
    env = gym.make(env_name)
    
    # Optimized discretization bins & limits
    if env_name == 'CartPole-v1':
        # Focus resolution on pole angle & angular velocity (72 states total)
        low = [-2.4, -3.0, -0.209, -3.0]
        high = [2.4, 3.0, 0.209, 3.0]
        bins = [1, 1, 6, 12]  
        near_optimal_threshold = 195.0
    else:  # MountainCar-v0
        # Position & velocity (252 states total)
        low = [-1.2, -0.07]
        high = [0.6, 0.07]
        bins = [18, 14]
        near_optimal_threshold = -110.0

    discretizer = StateDiscretizer(low, high, bins)
    
    if agent_type == 'Dyna-Q':
        agent = DynaQAgent(
            action_space_size=env.action_space.n,
            alpha=0.1,
            epsilon=0.1,
            n=n,
            gamma=0.99  # Higher discount factor helps long-horizon tasks
        )
    else:
        agent = PrioritizedSweepingAgent(
            action_space_size=env.action_space.n,
            alpha=0.1,
            epsilon=0.1,
            n=n,
            gamma=0.99
        )

    episodes_to_converge = None
    steps_per_episode_history = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state = discretizer.transform(obs)
        total_reward = 0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretizer.transform(next_obs)

            # CRITICAL FIX: Only pass `terminated` to the agent update!
            # Truncated steps must bootstrap future Q-values normally.
            agent.update(state, action, reward, next_state, terminated)

            state = next_state
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        steps_per_episode_history.append(steps)

        # Check convergence criteria
        if episodes_to_converge is None:
            if env_name == 'CartPole-v1' and total_reward >= near_optimal_threshold:
                episodes_to_converge = episode
            elif env_name == 'MountainCar-v0' and terminated and steps <= 150:
                episodes_to_converge = episode

    env.close()
    avg_steps_final = float(np.mean(steps_per_episode_history[-30:]))
    return episodes_to_converge, avg_steps_final