import pickle
from deepmdp.garage_mod.policies.discrete_qf_derived_policy import DiscreteQfDerivedPolicy
from deepmdp.experiments.dqn_baseline import setup_stacked_lunar_lander_env
import numpy as np
from gym.utils.play import play
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def evaluate_trained_policy(env):
    with open("configs/LunarLander/scalar-obs/baseline-new-architecture/deepmdp_baseline.pkl", "rb") as file:
        exp = pickle.load(file)
    algo = exp['algo']
    policy = DiscreteQfDerivedPolicy(
        env.spec,
        algo.qf
    )
    rewards_per_episode = []
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
        summed_reward = np.sum(rewards)
        rewards_per_episode.append(summed_reward)
        print(f"summed reward for this episode is {summed_reward}")
    print(f"Mean reward over {number_of_episodes} episodes was {np.mean(rewards_per_episode)}")


def play_env(setup_env, key_mapping=None):
    play(setup_env, keys_to_action=key_mapping)

def random_policy(env):
    max = 0
    min = 0
    for _ in range(number_of_episodes):
        obs = env.reset()
        rewards = []
        done = False
        while not done:
            act = env.action_space.sample()
            obs, reward, done, _ = env.step(act)
            rewards.append(reward)
            if np.max(obs) > max:
                max = np.max(obs)
            if np.min(obs) < min:
                min = np.min(obs)
        #print(f"summed reward for this episode is {np.sum(np.array(rewards))}")
    print(f"min {min} max {max}")

def show_lunar_obs(obs):
    obs = np.squeeze(obs)
    plt.imshow(obs, cmap='gray', vmin=0, vmax=255)
    plt.show()

lunar_lander_key_map = {(ord('w'), ): 2,
                   (ord('s'), ): 0,
                   (ord('d'), ): 3,
                   (ord('a'), ): 1}

if __name__=="__main__":
    number_of_episodes = 100
    env = setup_stacked_lunar_lander_env("LunarLander-v2", 4)
    #play_env(env, lunar_lander_key_map)
    evaluate_trained_policy(env)
