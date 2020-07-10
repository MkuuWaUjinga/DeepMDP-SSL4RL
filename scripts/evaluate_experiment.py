import pickle
import torch
from deepmdp.garage_mod.policies.discrete_qf_derived_policy import DiscreteQfDerivedPolicy
from deepmdp.experiments.dqn_baseline import setup_stacked_lunar_lander_env
import numpy as np
from gym.utils.play import play
from visdom import Visdom
from deepmdp.experiments.utils import VisdomLinePlotter
import matplotlib;

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gym
from garage.envs import GarageEnv

lunar_lander_key_map = {(ord('w'),): 2,
                        (ord('s'),): 0,
                        (ord('d'),): 3,
                        (ord('a'),): 1}

# Quantiles:
quantiles = np.array([[-0.84, 0.83],
             [-0.12, 1.6],
             [-1.38, 1.40],
             [-1.57, 0.36],
             [-1.41, 1.39],
             [-1.08, 0.89],
             [0, 1],
                      [0, 1]], dtype=np.float32)

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
            env.render()
            act = env.action_space.sample()
            obs, reward, done, _ = env.step(act)
            rewards.append(reward)
            if np.max(obs) > max:
                max = np.max(obs)
            if np.min(obs) < min:
                min = np.min(obs)
        # print(f"summed reward for this episode is {np.sum(np.array(rewards))}")
    print(f"min {min} max {max}")


def show_lunar_obs(obs):
    obs = np.squeeze(obs)
    plt.imshow(obs, cmap='gray', vmin=0, vmax=255)
    plt.show()


def normalize(vector):
    tiled_1th_quantile = np.tile(quantiles[:, 0], (num_stacked_frames, 1))
    tiled_99th_quantile = np.tile(quantiles[:, 1], (num_stacked_frames, 1))
    reshaped_normalized = (vector.reshape(8, 4).transpose() - tiled_1th_quantile) / (tiled_99th_quantile - tiled_1th_quantile)
    assert np.min(reshaped_normalized) >= 0 and np.max(reshaped_normalized) <= 1
    return reshaped_normalized.transpose().flatten()

def project_back(vector):
    tiled_1th_quantile = np.tile(quantiles[:, 0], (num_stacked_frames, 1))
    tiled_99th_quantile = np.tile(quantiles[:, 1], (num_stacked_frames, 1))
    reshaped_normalized = vector.reshape(8, 4).transpose() * (tiled_99th_quantile - tiled_1th_quantile) + tiled_1th_quantile
    return reshaped_normalized.transpose().flatten()

def latent_space_eval(obs, model_name, perturbed_index, viz):
    with open(f"configs/LunarLander/scalar-obs/baseline-new-architecture/{model_name}.pkl", "rb") as file:
        exp = pickle.load(file)
    algo = exp['algo']
    q_net = algo.qf
    q_net.eval()
    num_perturbations = 100
    l1_obs_list = []
    l1_embedding_list = []
    obs[perturbed_index*num_stacked_frames:(perturbed_index+1)*num_stacked_frames] = quantiles[perturbed_index][1]

    # Just for testing
    normalized_obs = normalize(obs)
    back_projected = project_back(normalized_obs)
    assert np.allclose(obs, back_projected, rtol=1e-6, atol=1e-6)

    with torch.no_grad():
        _, embedding = q_net(torch.Tensor(obs).unsqueeze(0), return_embedding=True)
#
    for j in range(num_perturbations):
        # perturb all occurences of attribute of index perturbed_index [-1, 1]
        normalized_obs = normalize(obs)
        perturbed_obs = np.copy(normalized_obs)
        perturbed_obs[perturbed_index*num_stacked_frames+3] = j/num_perturbations # Only perturb lates argument
        obs_delta = normalized_obs - perturbed_obs

        back_projected_perturbed_obs = project_back(perturbed_obs)
        with torch.no_grad():
            _, perturbed_embedding = q_net(torch.Tensor(back_projected_perturbed_obs).unsqueeze(0), return_embedding=True)
        embedding_delta = embedding - perturbed_embedding
        l1_obs_delta = np.linalg.norm(obs_delta, ord=1)
        l1_embedding_delta = np.linalg.norm(embedding_delta.numpy(), ord=1)
        l1_obs_list.append(l1_obs_delta.item())
        l1_embedding_list.append(l1_embedding_delta.item())

    x = np.stack((np.array(l1_obs_list), np.array(l1_embedding_list)), axis=1)
    x = x[x[:, 0].argsort()]
    for i in range(len(x)):
        viz.plot("l1 in embedding space", str(perturbed_index), "Latent Space Exp", x[i, 0], x[i, 1], color=np.array([[(int(255/6))*perturbed_index, (int(255/6))*(6-perturbed_index), 0], ]))
    #viz.line(
    #    X=x[:, 0],
    #    Y=x[:, 1],
    #    opts=dict(
    #        legend=list(range(6)),
    #       linecolor=np.array([[(int(255/6))*perturbed_index, (int(255/6))*(6-perturbed_index), 0], ])
    #    )
    #)


def determine_min_max(env, index):
    k = 100000
    done = True
    data = []
    for _ in range(k):
        if done:
            env.reset()
        act = env.action_space.sample()
        obs, _, done, env_info = env.step(act)
        val = obs[index]
        data.append(val)
    print(f"quantile 0.99 {np.quantile(np.array(data), 0.99)}")
    print(f"quantile 0.01 {np.quantile(np.array(data), 0.01)}")


if __name__ == "__main__":
    seed = 4
    num_stacked_frames = 4
    number_of_episodes = 100
    env = setup_stacked_lunar_lander_env("LunarLander-v2", num_stacked_frames)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_name_deepmdp = "deepmdp_baseline"
    model_name_vanilla = "vanilla_dqn_baseline_latent"

    # play_env(env, lunar_lander_key_map)

    viz = Visdom(port=9098)
    plotter = VisdomLinePlotter(viz, xlabel="l1 in observation space")
    obs = env.reset()
    for _ in range(4):
        obs, _, _, _ = env.step(0)
        env.render()

    for i in range(6):
        latent_space_eval(obs, model_name_vanilla, i, plotter)

