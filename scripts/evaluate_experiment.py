import pickle
from deepmdp.garage_mod.policies.discrete_qf_derived_policy import DiscreteQfDerivedPolicy
from deepmdp.experiments.dqn_baseline import setup_lunar_lander
import numpy as np
import random
from garage.envs import GarageEnv
from gym.utils.play import play
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from deepmdp.garage_mod.env_wrappers.lunar_lander_to_image_obs import LunarLanderToImageObservations

def evaluate_tained_policy(env):
    with open("runs/snapshots/params.pkl", "rb") as file:
        exp = pickle.load(file)
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

def play_env(setup_env, key_mapping=None):
    play(setup_env, keys_to_action=key_mapping)

def random_policy(env):
    for _ in range(number_of_episodes):
        obs = env.reset()
        rewards = []
        done = False
        while not done:
            act = env.action_space.sample()
            obs, reward, done, _ = env.step(act)
            rewards.append(reward)
            show_lunar_obs(obs)
        print(f"summed reward for this episode is {np.sum(np.array(rewards))}")

def show_lunar_obs(obs):
    obs = np.squeeze(obs)
    plt.imshow(obs, cmap='gray', vmin=0, vmax=255)
    plt.show()

lunar_lander_key_map = {(ord('w'), ): 2,
                   (ord('s'), ): 0,
                   (ord('a'), ): 3,
                   (ord('d'), ): 1}

if __name__=="__main__":
    number_of_episodes = 100
    env = setup_lunar_lander("LunarLander-v2")
    random_policy(env)