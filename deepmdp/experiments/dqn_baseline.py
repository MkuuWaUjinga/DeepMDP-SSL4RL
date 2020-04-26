import os
import sacred
import gym
import torch

from dowel import logger
from garage.experiment import SnapshotConfig
from garage.replay_buffer import SimpleReplayBuffer
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.grayscale import Grayscale
from garage.experiment.deterministic import get_seed
from garage.envs.wrappers.stack_frames import StackFrames
from garage.envs import GarageEnv

from deepmdp.garage_mod.off_policy_vectorized_sampler import OffPolicyVectorizedSampler
from deepmdp.garage_mod.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from deepmdp.garage_mod.policies.discrete_qf_derived_policy import DiscreteQfDerivedPolicy
from deepmdp.garage_mod.local_runner import LocalRunner
from deepmdp.garage_mod.algos.dqn import DQN
from deepmdp.garage_mod.algos.reward_auxiliary_objective import RewardAuxiliaryObjective
from deepmdp.garage_mod.env_wrappers.lunar_lander_to_image_obs import LunarLanderToImageObservations
from deepmdp.garage_mod.q_functions.discrete_cnn_q_function import DiscreteCNNQFunction

ex = sacred.experiment.Experiment("DQN-Baseline")

@ex.config
def config():
    snapshot_config = {"snapshot_dir": (os.path.join(os.getcwd(), 'runs/snapshots')),
                       "snapshot_mode": "last",
                       "snapshot_gap": 1}
    env_name = "SpaceInvaders-v0"
    dqn_config = {
        "replay_buffer_size": int(1e3),
        "n_epochs": 100,
        "steps_per_epoch": 20,
        "sampler_batch_size": 500,
        "n_train_steps": 500,
        "learning_rate": 0.0002,
        "buffer_batch_size": 32,
        "target_network_update_freq": 5,
        "min_buffer_size": 100,
        "net": {
            "filter_dims": (8, 4, 3),
            "num_filters": (32, 64, 64),
            "strides": (4, 2, 1),
            "dense_sizes": (512, ),
            "input_shape": (4, 84, 84)
        },
        "epsilon_greedy": {
            "max_epsilon": 1.0,
            "min_epsilon": 0.1,
            "decay_ratio": 0.1,
            "exponential_decay_rate": None, # Do linear decay with above params
            "episodical_decay": False
        }
    }


def setup_atari_env(env_name):
    env = gym.make(env_name)
    env = Noop(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    env = EpisodicLife(env)
    # Fire on reset as some envs are fixed until firing
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireReset(env)
    env = Grayscale(env)
    env = Resize(env, 84, 84)
    env = ClipReward(env)
    env = StackFrames(env, 4)
    return GarageEnv(env)

def setup_lunar_lander(env_name):
    env = gym.make(env_name)
    env = LunarLanderToImageObservations(env)
    env = Grayscale(env)
    env = Resize(env, 84, 84)
    env = StackFrames(env, 4)
    return GarageEnv(env)

def run_task(snapshot_config, env_name, dqn_config):
    logger.log(f"Config of this experiment is {dqn_config}")
    replay_buffer_size = dqn_config.get("replay_buffer_size")
    n_epochs = dqn_config.get("n_epochs")
    steps_per_epoch = dqn_config.get("steps_per_epoch")
    sampler_batch_size = dqn_config.get("sampler_batch_size")
    n_train_steps = dqn_config.get("n_train_steps")
    learning_rate = dqn_config.get("learning_rate")
    buffer_batch_size = dqn_config.get("buffer_batch_size")
    target_network_update_freq = dqn_config.get("target_network_update_freq")
    min_buffer_size = dqn_config.get("min_buffer_size")
    net_config = dqn_config.get("net")
    use_deepmdp = dqn_config.get("use_deepmdp")
    epsilon_greedy_config = dqn_config.get("epsilon_greedy")
    steps = n_epochs * steps_per_epoch * sampler_batch_size

    if "LunarLander-v2" in env_name:
        # Pass either LunarLander-v2 or LunarLander-v2-img to have the env give back image or semantical observations.
        if env_name[-4:] == "-img":
            env = setup_lunar_lander(env_name[:-4])
        else:
            env = GarageEnv(gym.make(env_name))
    else:
        env = setup_atari_env(env_name)

    # Set env seed
    env.seed(get_seed())
    # Set seed for action space (needed for epsilon-greedy reproducability)
    env.action_space.seed(get_seed())

    runner = LocalRunner(snapshot_config)
    replay_buffer = SimpleReplayBuffer(env.spec, size_in_transitions=replay_buffer_size, time_horizon=1)

    strategy = EpsilonGreedyStrategy(env.spec, steps, **epsilon_greedy_config)
    qf = DiscreteCNNQFunction(env_spec=env.spec,
                              **net_config)

    aux_objectives = []
    if use_deepmdp:
        reward_objective = RewardAuxiliaryObjective(env.spec, qf.embedding_size)
        aux_objectives.append(reward_objective)

    policy = DiscreteQfDerivedPolicy(env.spec, qf)
    algo = DQN(policy=policy,
               qf=qf,
               env_spec=env.spec,
               replay_buffer=replay_buffer,
               qf_optimizer=torch.optim.Adam,
               exploration_strategy=strategy,
               n_train_steps=n_train_steps,
               buffer_batch_size=buffer_batch_size,
               min_buffer_size=min_buffer_size,
               n_epoch_cycles=steps_per_epoch,
               target_network_update_freq=target_network_update_freq,
               qf_lr=learning_rate,
               max_path_length=1000,
               auxiliary_objectives=aux_objectives)

    # Use modded off policy sampler for passing generating summary statistics about episode's qvals in algo-object.
    algo.sampler_cls = OffPolicyVectorizedSampler

    runner.setup(algo=algo, env=env)
    runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)

    env.close()

@ex.main
def run(snapshot_config, env_name, dqn_config):
    snapshot_config = SnapshotConfig(snapshot_config["snapshot_dir"],
                                     snapshot_config["snapshot_mode"],
                                     snapshot_config["snapshot_gap"])
    run_task(snapshot_config, env_name, dqn_config)
