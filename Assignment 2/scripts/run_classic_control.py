import gymnasium as gym

def run_experiment(env_name, agent_type, n, num_episodes=500):
    env = gym.make(env_name)
    
    # Configure discretization boundaries based on environment
    if env_name == 'CartPole-v1':
        low = [-2.4, -3.0, -0.418, -3.0]
        high = [2.4, 3.0, 0.418, 3.0]
        bins = [6, 6, 12, 12]
        near_optimal_threshold = 195.0  # Avg reward over episode
    else:  # MountainCar-v0
        low = [-1.2, -0.07]
        high = [0.6, 0.07]
        bins = [20, 20]
        near_optimal_threshold = -110.0

    discretizer = StateDiscretizer(low, high, bins)
    
    if agent_type == 'Dyna-Q':
        agent = DynaQAgent(env.action_space.n, alpha=0.1, epsilon=0.1, n=n, gamma=0.9)
    else:
        agent = PrioritizedSweepingAgent(env.action_space.n, alpha=0.1, epsilon=0.1, n=n, gamma=0.9)

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
            done = terminated or truncated
            next_state = discretizer.transform(next_obs)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break

        steps_per_episode_history.append(steps)

        # Performance evaluation check
        if episodes_to_converge is None:
            if env_name == 'CartPole-v1' and total_reward >= near_optimal_threshold:
                episodes_to_converge = episode
            elif env_name == 'MountainCar-v0' and total_reward >= near_optimal_threshold:
                episodes_to_converge = episode

    env.close()
    avg_steps_final = np.mean(steps_per_episode_history[-50:])
    return episodes_to_converge, avg_steps_final