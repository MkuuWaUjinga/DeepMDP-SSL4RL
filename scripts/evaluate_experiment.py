import pickle
from deepmdp.garage_mod.policies.discrete_qf_derived_policy import DiscreteQfDerivedPolicy
from deepmdp.experiments.dqn_baseline import setup_atari_env
import numpy as np
import random
from garage.envs import GarageEnv
from gym.utils import play

number_of_episodes = 100
with open("runs/snapshots/params.pkl", "rb") as file:
    exp = pickle.load(file)
env = GarageEnv(env_name="LunarLander-v2")
algo = exp['algo']
policy = DiscreteQfDerivedPolicy(
    env.spec,
    algo.qf
)

for _ in range(number_of_episodes):
    obs = env.reset()
    rewards = []
    env.render()
    done = False
    while not done:
        act, _ = policy.get_action(obs)
        obs, reward, done, _ = env.step(act.numpy())
        rewards.append(reward)
        env.render()
    print(f"summed reward for this episode is {np.sum(np.array(rewards))}")


def play(setup_env):
    play(setup_env)
