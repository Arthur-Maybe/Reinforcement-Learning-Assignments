import numpy as np

class StateDiscretizer:
    def __init__(self, low, high, num_bins):
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.num_bins = num_bins

        # Build bin boundaries for each state component
        self.bins = [
                np.linspace(l, h, b + 1)[1:-1]
                for l, h, b in zip(self.low, self.high, self.num_bins)
        ]

    def transform(self, state):
        """
        Discretizes the continuous state into discrete bins.

        Parameters:
        - state: A numpy array representing the continuous state.

        Returns:
        - A tuple of integers representing the discretized state.
        """
        state = np.clip(state, self.low, self.high)  # Ensure state is within bounds
        # Find index for each dimension
        indices = [
            int(np.digitize(s, bin_edges))
            for s, bin_edges in zip(state, self.bins)
        ]
        return tuple(indices)