import numpy as np
import gym
import gym.spaces
from gym.envs.box2d.lunar_lander import VIEWPORT_H, VIEWPORT_W

class LunarLanderToImageObservations(gym.Wrapper):
    """Wrapper for Lunar Lander environments. Turns LunarLander's semantically rich observation vector to RGB-image
    observations.
    This wrapper only works if you use it as the first wrapper for a gym env

    Example:
        env = gym.make('LunarLander-v2')
        env_wrapped = LunarLanderToImageObservations(env)

    Args:
        env: Lunar Lander gym.Env to wrap.

    """

    def __init__(self, env):
        assert env.spec.id == 'LunarLander-v2'
        super().__init__(env)
        self._observation_space = gym.spaces.Box(
            0,
            255,
            shape=(VIEWPORT_W, VIEWPORT_H, 3),
            dtype=np.uint8)

    @property
    def observation_space(self):
        """gym.Env observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def render_image(self):
        return self.env.render(mode="rgb_array")

    def reset(self) -> int:
        """gym.Env reset function."""
        self.env.reset()
        return self.render_image()

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        info["ground_truth_state"] = obs
        return self.render_image(), reward, done, info