import random
import heapq
from collections import defaultdict
import numpy as np

class PrioritizedSweepingAgent:
    def __init__(self, action_space_size, alpha=0.1, epsilon=0.1, n=10, gamma=0.9, theta=1e-4):
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        self.gamma = gamma
        self.theta = theta

        self.Q = defaultdict(lambda: np.zeros(self.action_space_size))
        self.model = {}
        # Predecessors map: predecessors[next_state] = set of (state, action)
        self.predecessors = defaultdict(set)
        
        # Max-priority queue storing (-priority, state, action)
        self.pq = []
        self.pq_set = set()  # To avoid duplicate tracking in queue

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        return int(np.argmax(self.Q[state]))

    def _push_to_pq(self, priority, state, action):
        if priority > self.theta:
            # Avoid duplicate entries in heap
            if (state, action) not in self.pq_set:
                heapq.heappush(self.pq, (-priority, state, action))
                self.pq_set.add((state, action))

    def update(self, state, action, reward, next_state, done):
        # 1. Update model & predecessors
        self.model[(state, action)] = (reward, next_state, done)
        self.predecessors[next_state].add((state, action))

        # 2. Priority calculation for current transition
        best_next_q = np.max(self.Q[next_state]) if not done else 0.0
        p = abs(reward + self.gamma * best_next_q - self.Q[state][action])
        self._push_to_pq(p, state, action)

        # 3. Planning loop
        count = 0
        while self.pq and count < self.n:
            neg_p, s_sim, a_sim = heapq.heappop(self.pq)
            self.pq_set.remove((s_sim, a_sim))

            r_sim, next_s_sim, done_sim = self.model[(s_sim, a_sim)]
            
            # Q-update
            best_next_q_sim = np.max(self.Q[next_s_sim]) if not done_sim else 0.0
            self.Q[s_sim][a_sim] += self.alpha * (
                r_sim + self.gamma * best_next_q_sim - self.Q[s_sim][a_sim]
            )
            count += 1

            # Check predecessors of s_sim
            for s_pre, a_pre in self.predecessors[s_sim]:
                r_pre, _, _ = self.model[(s_pre, a_pre)]
                best_q_s_sim = np.max(self.Q[s_sim])
                p_pre = abs(r_pre + self.gamma * best_q_s_sim - self.Q[s_pre][a_pre])
                self._push_to_pq(p_pre, s_pre, a_pre)