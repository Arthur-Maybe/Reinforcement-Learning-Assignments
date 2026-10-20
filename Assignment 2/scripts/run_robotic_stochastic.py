import numpy as np
from envs.robotic_manipulation_nd import RobotArmNDEnv
from agents.stochastic_dyna_q import StochasticDynaQAgent

def train_stochastic_arm(n_planning_steps, num_episodes=400):
    env = RobotArmNDEnv()
    agent = StochasticDynaQAgent(
        action_space_size=env.action_space.n,
        alpha=0.1,
        epsilon=0.1,
        n=n_planning_steps,
        gamma=0.9
    )

    episodes_to_converge = None
    steps_history = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state = env.obs_to_state_tuple(obs)
        steps = 0
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = env.obs_to_state_tuple(next_obs)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        steps_history.append(steps)

        if total_reward > 0 and episodes_to_converge is None:
            episodes_to_converge = episode

    final_avg_steps = np.mean(steps_history[-30:])
    return episodes_to_converge, final_avg_steps

if __name__ == "__main__":
    n_values = [0, 5, 10, 20, 50]
    print("--- Part (b)(ii): Non-Deterministic Robotic Arm Domain ---")
    for n in n_values:
        ep, steps = train_stochastic_arm(n)
        print(f"n = {n:2d} | Stochastic Dyna-Q: Converged in {ep} episodes | Avg Steps: {steps:.1f}")