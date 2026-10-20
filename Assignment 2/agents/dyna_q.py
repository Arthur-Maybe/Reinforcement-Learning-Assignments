import random
from collections import defaultdict
import numpy as np

class DynaQAgent:
    def __init__(self, action_space_size, alpha=0.1, epsilon=0.1, n=10, gamma=0.9):
        self.action_space_size = action_space_size
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.n = n  # Number of planning steps
        self.gamma = gamma  # Discount factor

        # Initialize Q-table as a defaultdict of defaultdicts
        self.Q = defaultdict(lambda: np.zeros(self.action_space_size))

        # Model for Dyna-Q: stores transitions (state, action) -> (next_state, reward)
        self.model = {}
        # Set of observed state-action pairs
        self.observed_sa = set()

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.

        Parameters:
        - state: The current state.

        Returns:
        - An integer representing the selected action.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)  # Explore
        else:
            return int(np.argmax(self.Q[state]))  # Exploit

    def update(self, state, action, reward, next_state, done):
        """
        Updates the Q-table and the model based on the observed transition.

        Parameters:
        - state: The current state.
        - action: The action taken.
        - reward: The reward received.
        - next_state: The next state after taking the action.
        - done: A boolean indicating if the episode has ended.
        """
        # Update Q-value using the Q-learning update rule
        best_next_q = np.max(self.Q[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        # Update the model with the observed transition
        self.model[(state, action)] = (reward, next_state, done)
        self.observed_sa.add((state, action))

        # Perform planning steps
        if self.n > 0 and len(self.observed_sa) > 0:
            observed_list = list(self.observed_sa)
            for _ in range(self.n):
                s_sim, a_sim = random.choice(observed_list)
                r_sim, next_s_sim, done_sim = self.model[(s_sim, a_sim)]
                
                best_next_q_sim = np.max(self.Q[next_s_sim]) if not done_sim else 0.0
                td_target_sim = r_sim + self.gamma * best_next_q_sim
                self.Q[s_sim][a_sim] += self.alpha * (td_target_sim - self.Q[s_sim][a_sim])

                