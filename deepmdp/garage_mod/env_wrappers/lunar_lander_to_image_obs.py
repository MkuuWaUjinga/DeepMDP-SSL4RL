import numpy as np
import pyglet
from pyglet.gl import *
import gym
import gym.spaces
from gym.envs.classic_control.rendering import Viewer, Transform
from gym.envs.box2d.lunar_lander import VIEWPORT_H, VIEWPORT_W, SCALE

class LunarLanderToImageObservations(gym.Wrapper):
    """Wrap for Lunar Lander environments. Turns LunarLander's semantically rich observation vector to RGB-image
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

    def convert_to_image(self, obs):
        # Inject custom viewer in Lunar Lander env that doesn't open a window.
        #viewer = LunarLanderViewer(VIEWPORT_W, VIEWPORT_H)
        #viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)
        #env_viewer = self.env.env.viewer
        #self.env.env.viewer = viewer
        obs = self.env.render(mode="rgb_array")
        #self.env.env.viewer = env_viewer
        return obs

    def reset(self):
        """gym.Env reset function."""
        return self.convert_to_image(self.env.reset())

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return self.convert_to_image(obs), reward, done, info

class LunarLanderViewer(Viewer):

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

    def close(self):
        pass

    def render(self, return_rgb_array=False):
        #window = pyglet.window.Window(width=self.width, height=self.height, visible=True)
        #window.minimize()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(1, 1, 1, 1)
        #window.clear()
        #window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.data, dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        #window.flip()
        self.onetime_geoms = []
        #window.close()
        return arr