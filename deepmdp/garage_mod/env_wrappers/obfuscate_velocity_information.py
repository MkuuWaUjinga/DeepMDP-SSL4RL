import numpy as np
import gym
import gym.spaces

class ObfuscateVelocityInformation(gym.Wrapper):
    """Wrapper for Lunar Lander environments. Obfuscates velocity information from LunarLander's semantically rich
    observation vector. Namely the lander's velocity in x and y direction as well as the angular velocity.

    Example:
        env = gym.make('LunarLander-v2')
        env_wrapped = ObfuscateVelocityInformation(env)

    Args:
        env: Lunar Lander gym.Env to wrap.
    """

    state_meaning_to_index = {
        "pos_x": 0,
        "pos_y": 1,
        "velocity_x": 2,
        "velocity_y": 3,
        "angle": 4,
        "angular_velocity": 5,
        "leg_one_has_ground_contact": 6,
        "leg_two_has_ground_contact": 7
    }

    def __init__(self, env):
        assert env.spec.id == 'LunarLander-v2'
        super().__init__(env)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self._state_selection_mask = np.ones(8, dtype=bool)
        self._state_selection_mask[[self.state_meaning_to_index["velocity_x"],
                                    self.state_meaning_to_index["velocity_y"],
                                    self.state_meaning_to_index["angular_velocity"]]] = False

    @property
    def observation_space(self):
        """gym.Env observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def _observation(self, obs):
        return obs[self._state_selection_mask]

    def reset(self):
        """gym.Env reset function."""
        return self._observation(self.env.reset())

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        info["obfuscated_state_info"] = obs[~self._state_selection_mask]
        return self._observation(obs), reward, done, info