import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RobotArmEnv(gym.Env):
    """
    Custom Robotic Arm Environment for Pick-and-Place Task (Deterministic)
    """
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, render_mode=None, max_steps=50):
        super().__init__()

        # Link lengths
        self.d1 = 1.0
        self.d2 = 1.5

        # Bolt pickup (xb, yb) and hole target (xh, yh)
        self.bolt_pos = np.array([2.176, 0.129])
        self.hole_pos = np.array([2.266, 0.643])

        # Joint Limits (in radians)
        self.min_theta1, self.max_theta1 = 0.0, np.pi / 3.0
        self.min_theta2, self.max_theta2 = -np.pi / 3.0, np.pi / 3.0
        self.detal_theta = np.pi / 9.0  # Increment for joint angles

        # Action Space: MultiDiscrete([3, 3, 2]) mapped to discrete int 0..17
        self.action_space = spaces.Discrete(18)

        # Observation Space
        low = np.array([self.min_theta1, self.min_theta2, 0, 0], dtype=np.float32)
        high = np.array([self.max_theta1, self.max_theta2, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.reset()

    @staticmethod
    def action_int_to_tuple(a_int):
        """Converts integer action 0..17 to MultiDiscrete [d_t1, d_t2, gripper]"""
        a0 = a_int // 6
        a1 = (a_int // 2) % 3
        a2 = a_int % 2
        return np.array([a0, a1, a2], dtype=int)

    @staticmethod
    def obs_to_state_tuple(obs):
        """Converts continuous observation array to a discrete tuple key for Q-table"""
        return (round(float(obs[0]), 4), round(float(obs[1]), 4), int(obs[2]), int(obs[3]))

    def _get_obs(self):
        return np.array([self.theta1, self.theta2, float(self.gripper_state), float(self.has_bolt)], dtype=np.float32)

    def _get_gripper_pos(self):
        xg = self.d1 * np.cos(self.theta1) + self.d2 * np.cos(self.theta1 + self.theta2)
        yg = self.d1 * np.sin(self.theta1) + self.d2 * np.sin(self.theta1 + self.theta2)
        return np.array([xg, yg])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.theta1 = np.pi / 9
        self.theta2 = np.pi / 9
        self.gripper_state = 0  # 0: open, 1: closed
        self.has_bolt = 0       # 0: no bolt, 1: has bolt
        self.current_step = 0

        observation = self._get_obs()
        return observation, {}

    def step(self, action_int):
        self.current_step += 1
        
        # Decode integer action to discrete choices
        action = self.action_int_to_tuple(action_int)
        d_theta1 = (action[0] - 1) * self.detal_theta  # -20, 0, +20 degrees
        d_theta2 = (action[1] - 1) * self.detal_theta
        new_gripper = action[2]

        # Update joint angles with limits
        self.theta1 = np.clip(self.theta1 + d_theta1, self.min_theta1, self.max_theta1)
        self.theta2 = np.clip(self.theta2 + d_theta2, self.min_theta2, self.max_theta2)
        self.gripper_state = new_gripper

        gripper_pos = self._get_gripper_pos()
        pickup_tolerance = 0.15

        # Handle bolt pickup & drop
        if self.has_bolt == 0 and self.gripper_state == 1:
            if np.linalg.norm(gripper_pos - self.bolt_pos) <= pickup_tolerance:
                self.has_bolt = 1

        if self.has_bolt == 1 and self.gripper_state == 0:
            self.has_bolt = 0

        # Evaluate terminal conditions and rewards
        reward = -1.0
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Successful placement condition
        if self.has_bolt == 1 and np.linalg.norm(gripper_pos - self.hole_pos) <= pickup_tolerance:
            reward = 10.0
            terminated = True
        # Collision condition
        elif (gripper_pos[0] >= self.hole_pos[0] or gripper_pos[1] <= 0.0) and not (np.linalg.norm(gripper_pos - self.hole_pos) <= pickup_tolerance):
            reward = -100.0
            terminated = True

        observation = self._get_obs()
        return observation, reward, terminated, truncated, {"gripper_pos": gripper_pos}