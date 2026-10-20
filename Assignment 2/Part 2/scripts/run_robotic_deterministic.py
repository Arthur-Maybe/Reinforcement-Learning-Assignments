import numpy as np
from envs.robotic_manipulation import RobotArmEnv
from agents.dyna_q import DynaQAgent
from agents.prioritized_sweeping import PrioritizedSweepingAgent

def train_arm_agent(agent_class, n_planning_steps, num_episodes=300):
    env = RobotArmEnv()
    agent = agent_class(
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

        # Convergence criterion: Reaching the goal in minimum steps (reward = +10)
        if total_reward > 0 and episodes_to_converge is None:
            episodes_to_converge = episode

    final_avg_steps = np.mean(steps_history[-30:])
    return episodes_to_converge, final_avg_steps

if __name__ == "__main__":
    n_values = [0, 5, 10, 20, 50]
    print("--- Part (b)(i): Deterministic Robotic Arm Domain ---")
    
    for n in n_values:
        ep_dq, steps_dq = train_arm_agent(DynaQAgent, n)
        ep_ps, steps_ps = train_arm_agent(PrioritizedSweepingAgent, n)
        print(f"n = {n:2d} | Dyna-Q: {ep_dq} episodes, {steps_dq:.1f} steps/ep | Prioritized Sweeping: {ep_ps} episodes, {steps_ps:.1f} steps/ep")