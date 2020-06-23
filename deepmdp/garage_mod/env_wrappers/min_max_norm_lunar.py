import gym
import numpy as np

class MinMaxNormLunar(gym.Wrapper):
    """Wrapper for Lunar Lander environments. Normalizes rewards to [-1, 1].
    """
    def reset(self) -> int:
        """gym.Env reset function."""
        return self.env.reset()

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return self.min_max_norm(obs, -8, 8), reward, done, info

    @staticmethod
    def min_max_norm(x, min, max):
        """
        Normalize array to [-1, 1] range
        """
        x = np.clip(x, min, max) # Just a safeguard
        return - 1 + (x - min) * 2 / (max - min)