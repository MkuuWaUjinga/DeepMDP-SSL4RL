import os
import sacred
import gym
import torch
import numpy as np

from garage.np.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from garage.torch.algos.ddpg import DDPG
from garage.experiment import SnapshotConfig
from garage.replay_buffer import SimpleReplayBuffer
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames
from garage.envs import GarageEnv, normalize
from deepmdp.garage_mod.env_wrappers.grayscale import Grayscale
from deepmdp.garage_mod.policies.discrete_qf_derived_policy import DiscreteQfDerivedPolicy
from deepmdp.garage_mod.local_runner import LocalRunner
from deepmdp.garage_mod.algos.dqn import DQN
from deepmdp.garage_mod.q_functions.discrete_cnn_q_function import DiscreteCNNQFunction

ex = sacred.experiment.Experiment("DQN-Baseline")

@ex.config
def config():
    snapshot_config = {"snapshot_dir": (os.path.join(os.getcwd(), 'runs/snapshots')),
                       "snapshot_mode": "last",
                       "snapshot_gap": 1}
    env_name = "SpaceInvaders-v0"

def run_task(snapshot_config, env_name):
    n_epochs = 400
    steps_per_epoch = 500
    sampler_batch_size = 500
    num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size


    env = (gym.make(env_name))
    env = Noop(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    env = EpisodicLife(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        # Does that mean that we should always fire if there is the action to do so?
        env = FireReset(env)
    env = Grayscale(env)
    env = Resize(env, 84, 84)
    # TODO check whether reward should be continuous between -1 and 1
    env = ClipReward(env)
    # Create Game State
    env = StackFrames(env, 4)
    env = GarageEnv(env)

    runner = LocalRunner(snapshot_config)
    replay_buffer = SimpleReplayBuffer(env.spec, size_in_transitions=int(1e7), time_horizon=1)

    # Todo check whether it is decaying linearly or exponentially
    strategy = EpsilonGreedyStrategy(env.spec, num_timesteps, max_epsilon=1, min_epsilon=0.1)

    qf = DiscreteCNNQFunction(env_spec=env.spec,
                              filter_dims=(4, 4, 4, 3),
                              num_filters=(32, 32, 64, 64),
                              strides=(2, 2, 1, 1),
                              dense_sizes=(256, 256),
                              input_shape=(4, 84, 84))
    policy = DiscreteQfDerivedPolicy(env.spec, qf)

    algo = DQN(policy=policy,
               qf=qf,
               env_spec=env.spec,
               replay_buffer=replay_buffer,
               qf_optimizer=torch.optim.Adam,
               exploration_strategy=strategy)

    runner.setup(algo=algo, env=env)
    runner.train(n_epochs=400, batch_size=100)

@ex.main
def run(snapshot_config, env_name):
    snapshot_config = SnapshotConfig(snapshot_config["snapshot_dir"],
                                     snapshot_config["snapshot_mode"],
                                     snapshot_config["snapshot_gap"])
    run_task(snapshot_config, env_name)
