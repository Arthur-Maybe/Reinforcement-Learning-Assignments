import random
from collections import defaultdict
import numpy as np

class StochasticDynaQAgent:
    def __init__(self, action_space_size, alpha=0.1, epsilon=0.1, n=10, gamma=0.9):
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        self.gamma = gamma

        self.Q = defaultdict(lambda: np.zeros(self.action_space_size))
        
        # Transition counts: N_sas[state][action][next_state] = count
        self.N_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # Total state-action counts: N_sa[state][action] = total
        self.N_sa = defaultdict(lambda: defaultdict(int))
        # Total reward accumulation: R_sa[state][action] = sum_r
        self.R_sa = defaultdict(lambda: defaultdict(float))
        # Terminal state indicator: Done_sas[state][action][next_state] = bool
        self.Done_sas = {}

        self.observed_sa = set()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        # 1. Direct Q-learning update
        best_next_q = np.max(self.Q[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next_q
        self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])

        # 2. Count-based Model Update
        self.N_sas[state][action][next_state] += 1
        self.N_sa[state][action] += 1
        self.R_sa[state][action] += reward
        self.Done_sas[(state, action, next_state)] = done
        self.observed_sa.add((state, action))

        # 3. Planning steps with Stochastic Model
        if self.n > 0 and len(self.observed_sa) > 0:
            observed_list = list(self.observed_sa)
            for _ in range(self.n):
                s_sim, a_sim = random.choice(observed_list)
                
                # Calculate expected reward
                r_sim = self.R_sa[s_sim][a_sim] / self.N_sa[s_sim][a_sim]
                
                # Sample next state based on empirical distribution
                next_states = list(self.N_sas[s_sim][a_sim].keys())
                counts = list(self.N_sas[s_sim][a_sim].values())
                probs = [c / self.N_sa[s_sim][a_sim] for c in counts]
                
                next_s_sim = random.choices(next_states, weights=probs, k=1)[0]
                done_sim = self.Done_sas[(s_sim, a_sim, next_s_sim)]

                best_next_q_sim = np.max(self.Q[next_s_sim]) if not done_sim else 0.0
                td_target_sim = r_sim + self.gamma * best_next_q_sim
                self.Q[s_sim][a_sim] += self.alpha * (td_target_sim - self.Q[s_sim][a_sim])