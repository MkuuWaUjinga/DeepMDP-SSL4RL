import os
import sacred
import gym
import torch

from garage.np.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from garage.experiment import SnapshotConfig
from garage.replay_buffer import SimpleReplayBuffer
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames
from garage.envs import GarageEnv
from deepmdp.garage_mod.env_wrappers.grayscale import Grayscale
from deepmdp.garage_mod.policies.discrete_qf_derived_policy import DiscreteQfDerivedPolicy
from deepmdp.garage_mod.local_runner import LocalRunner
from deepmdp.garage_mod.algos.dqn import DQN
from deepmdp.garage_mod.q_functions.discrete_cnn_q_function import DiscreteCNNQFunction
from dowel import logger

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
        "learning_rate": 0.0002,
        "buffer_batch_size": 32
    }


def setup_atari_env(env_name):
    env = gym.make(env_name)
    env = Noop(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    env = EpisodicLife(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireReset(env)
    env = Grayscale(env)
    env = Resize(env, 84, 84)
    # TODO check whether reward should be continuous between -1 and 1 Hitting a mothership gives more points than hitting a normal alien!
    env = ClipReward(env)
    # Create Game State
    env = StackFrames(env, 4)
    return GarageEnv(env)

def is_atari(env_name):
    atari_env_names = ["SpaceInvaders-v0"]
    return env_name in atari_env_names

def run_task(snapshot_config, env_name, dqn_config):
    logger.log(f"Config of this experiment is {dqn_config}")
    replay_buffer_size = dqn_config.get("replay_buffer_size")
    n_epochs = dqn_config.get("n_epochs")
    steps_per_epoch = dqn_config.get("steps_per_epoch")
    sampler_batch_size = dqn_config.get("sampler_batch_size")
    learning_rate = dqn_config.get("learning_rate")
    buffer_batch_size = dqn_config.get("buffer_batch_size")
    steps = n_epochs * steps_per_epoch * sampler_batch_size
    n_train_steps = sampler_batch_size

    if not is_atari(env_name):
        env = GarageEnv(env_name="LunarLander-v2")
    else:
        env = setup_atari_env(env_name)

    runner = LocalRunner(snapshot_config)
    replay_buffer = SimpleReplayBuffer(env.spec, size_in_transitions=replay_buffer_size, time_horizon=1)

    # Todo check whether epsilon is decaying linearly
    strategy = EpsilonGreedyStrategy(env.spec, steps, max_epsilon=1, min_epsilon=0.1)

    qf = DiscreteCNNQFunction(env_spec=env.spec,
                              filter_dims=(),
                              num_filters=(),
                              strides=(),
                              dense_sizes=(512, 256),
                              input_shape=(8,))
    print(qf)
    policy = DiscreteQfDerivedPolicy(env.spec, qf)
    algo = DQN(policy=policy,
               qf=qf,
               env_spec=env.spec,
               replay_buffer=replay_buffer,
               qf_optimizer=torch.optim.Adam,
               exploration_strategy=strategy,
               n_train_steps=n_train_steps,
               buffer_batch_size=buffer_batch_size,
               min_buffer_size=100,
               n_epoch_cycles=steps_per_epoch,
               qf_lr=learning_rate)
    runner.setup(algo=algo, env=env)
    runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)

@ex.main
def run(snapshot_config, env_name, dqn_config):
    snapshot_config = SnapshotConfig(snapshot_config["snapshot_dir"],
                                     snapshot_config["snapshot_mode"],
                                     snapshot_config["snapshot_gap"])
    run_task(snapshot_config, env_name, dqn_config)
