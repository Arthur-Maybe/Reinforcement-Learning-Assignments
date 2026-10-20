import random
import numpy as np
from envs.robotic_manipulation import RobotArmEnv

class RobotArmNDEnv(RobotArmEnv):
    """
    Non-Deterministic Robotic Arm Environment ("robotic manipulation nd")
    - 0.9 probability: intended joint angle transition succeeds
    - 0.1 probability: action fails, resulting in no state change
    """
    def step(self, action_int):
        # 10% chance that action fails -> no state change
        if random.random() < 0.1:
            self.current_step += 1
            reward = -1.0
            truncated = self.current_step >= self.max_steps
            gripper_pos = self._get_gripper_pos()
            return self._get_obs(), reward, False, truncated, {"gripper_pos": gripper_pos}

        # 90% chance that action succeeds normally
        return super().step(action_int)