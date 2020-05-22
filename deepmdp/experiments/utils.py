import numpy as np
from visdom import Visdom

def show_agent_playing(policy, env):
    obs = env.reset()
    env.render()
    for i in range(1000):
        action = policy.get_actions(obs)
        print(action[0])
        obs, rewards, done, env_infos = env.step(action[0])
        if done:
            break
        env.render()


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, port, env_name='main', xlabel='Iteration'):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.plots = {}
        self.xlabel = xlabel

    def plot(self, var_name, split_name, title_name, x, y, color=None):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                title=title_name,
                xlabel=self.xlabel,
                ylabel=var_name,
                linecolor=color
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append', opts=dict(linecolor=color))
