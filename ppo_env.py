# ppo_env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AICPPOEnv(gym.Env):
    """
    First PPO environment for AIC.

    This is NOT the final competition policy.
    This is the training wrapper:
        reset() -> start episode
        step(action) -> move robot, read obs, compute reward
    """

    def __init__(self, ros_interface, max_steps=120):
        super().__init__()

        self.ros = ros_interface
        self.max_steps = max_steps
        self.step_count = 0
        self.prev_distance = None

        # Action: tiny Cartesian delta in TCP/base frame.
        # PPO outputs values in [-1, 1].
        # We scale them to millimeters.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # Observation vector:
        # [tcp_x, tcp_y, tcp_z,
        #  tcp_vx, tcp_vy, tcp_vz,
        #  force_x, force_y, force_z,
        #  target_dx, target_dy, target_dz]
        #
        # For first training, target_dx/dy/dz can come from ground truth.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        # Later this should respawn/randomize the scene.
        self.ros.reset_episode()

        obs = self.ros.get_observation_vector()
        distance = self.ros.get_plug_to_port_distance()

        self.prev_distance = distance

        return obs.astype(np.float32), {}

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)

        # Scale PPO action to small robot movement.
        dx = float(action[0]) * 0.003  # 3 mm
        dy = float(action[1]) * 0.003  # 3 mm
        dz = float(action[2]) * 0.002  # 2 mm

        self.ros.move_tcp_delta(dx, dy, dz)

        obs = self.ros.get_observation_vector()

        distance = self.ros.get_plug_to_port_distance()
        force = self.ros.get_force_norm()
        inserted = self.ros.is_inserted()

        reward = self._compute_reward(distance, force, inserted)

        terminated = inserted
        truncated = self.step_count >= self.max_steps

        self.prev_distance = distance

        info = {
            "distance": distance,
            "force": force,
            "inserted": inserted,
        }

        return obs.astype(np.float32), reward, terminated, truncated, info

    def _compute_reward(self, distance, force, inserted):
        reward = 0.0

        # Reward improvement toward the port.
        if self.prev_distance is not None:
            reward += 10.0 * (self.prev_distance - distance)

        # Reward being close.
        reward += -1.0 * distance

        # Big reward for success.
        if inserted:
            reward += 10.0

        # Penalize excessive force.
        if force > 20.0:
            reward -= 5.0

        # Small step penalty so it learns to finish.
        reward -= 0.01

        return reward