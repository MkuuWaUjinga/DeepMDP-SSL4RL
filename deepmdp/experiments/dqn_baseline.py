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
from deepmdp.garage_mod.env_wrappers.min_max_norm_lunar import MinMaxNormLunar
from deepmdp.garage_mod.env_wrappers.obfuscate_velocity_information import ObfuscateVelocityInformation
from deepmdp.garage_mod.q_functions.discrete_cnn_q_function import DiscreteCNNQFunction
from deepmdp.garage_mod.algos.transition_auxiliary_objective import TransitionAuxiliaryObjective
from deepmdp.experiments.utils import Visualizer

ex = sacred.experiment.Experiment("DQN-Baseline")
num_frames = 4

@ex.capture
def get_info(_run):
    return(_run._id)


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
    env = StackFrames(env, num_frames)
    return GarageEnv(env)

def setup_lunar_lander_with_image_obs(env_name, do_noops=False):
    from deepmdp.garage_mod.env_wrappers.stack_frames import StackFrames
    env = gym.make(env_name)
    env = LunarLanderToImageObservations(env)
    env = Grayscale(env)
    env = Resize(env, 84, 84)
    env = StackFrames(env, num_frames, do_noops=do_noops)
    return GarageEnv(env)

def setup_lunar_lander_with_obfuscated_states(env_name, number_of_stacked_frames=4, do_noops=False):
    from deepmdp.garage_mod.env_wrappers.stack_frames import StackFrames
    env = gym.make(env_name)
    env = ObfuscateVelocityInformation(env)
    env = StackFrames(env, number_of_stacked_frames, do_noops=do_noops)
    return GarageEnv(env)

def setup_stacked_lunar_lander_env(env_name, normalize=False):
    from deepmdp.garage_mod.env_wrappers.stack_frames import StackFrames
    env = gym.make(env_name)
    env = ObfuscateVelocityInformation(env, no_obf=True)
    if normalize:
        env = MinMaxNormLunar(env)
    env = StackFrames(env, num_frames)
    return GarageEnv(env)

def run_task(snapshot_config, exp_config):
    logger.log(f"Config of this experiment is {exp_config}")
    env_config = exp_config["env"]
    env_name = env_config["name"]
    replay_buffer_size = exp_config.get("replay_buffer_size")
    n_epochs = exp_config.get("n_epochs")
    steps_per_epoch = exp_config.get("steps_per_epoch")
    sampler_batch_size = exp_config.get("sampler_batch_size")
    n_train_steps = exp_config.get("n_train_steps")
    learning_rate = exp_config.get("learning_rate")
    buffer_batch_size = exp_config.get("buffer_batch_size")
    target_network_update_freq = exp_config.get("target_network_update_freq")
    min_buffer_size = exp_config.get("min_buffer_size")
    net_config = exp_config.get("q-net")
    loss_weights = exp_config.get("loss_weights")
    deepmdp_config = exp_config.get("deepmdp")
    epsilon_greedy_config = exp_config.get("epsilon_greedy")
    plots = exp_config.get("plots")
    steps = n_epochs * steps_per_epoch * sampler_batch_size
    snapshot_config = SnapshotConfig(os.path.join(os.getcwd(), f'runs/{get_info()}/snapshots'),
                                     snapshot_config["snapshot_mode"],
                                     snapshot_config["snapshot_gap"])

    if "LunarLander-v2" in env_name:
        # Pass either LunarLander-v2 or LunarLander-v2-img to have the env give back image or semantical observations.
        if env_name[-4:] == "-img":
            env = setup_lunar_lander_with_image_obs(env_name[:-4], do_noops=env_config["do_noops"])
        elif env_name[-4:] == "-obf":
            env = setup_lunar_lander_with_obfuscated_states(env_name[:-4], do_noops=env_config["do_noops"])
        elif env_name[-4:] == "-stk":
            env = setup_stacked_lunar_lander_env(env_name[:-4], normalize=env_config["normalize"])
        else:
            env = GarageEnv(gym.make(env_name))
    elif "SpaceInvaders-v0" == env_name:
        env = setup_atari_env(env_name)
    else:
        raise ValueError("Env name not known")

    # Set env seed
    env.seed(get_seed())
    # Set seed for action space (needed for epsilon-greedy reproducability)
    env.action_space.seed(get_seed())

    # Init visualizer
    visualizer = Visualizer(get_info() + "_main", plots)
    visualizer.publish_config(exp_config)

    runner = LocalRunner(snapshot_config)
    replay_buffer = SimpleReplayBuffer(env.spec, size_in_transitions=replay_buffer_size, time_horizon=1)

    strategy = EpsilonGreedyStrategy(env.spec, steps, **epsilon_greedy_config)
    qf = DiscreteCNNQFunction(env_spec=env.spec,
                              **net_config)

    aux_objectives = []
    if deepmdp_config["use"]:
        reward_objective = RewardAuxiliaryObjective(env.spec, qf.embedding_size, deepmdp_config["reward_head"])
        transition_objective = TransitionAuxiliaryObjective(env.spec, qf.embedding_size, deepmdp_config["transition_head"])
        aux_objectives.append(reward_objective)
        aux_objectives.append(transition_objective)

    policy = DiscreteQfDerivedPolicy(env.spec, qf)
    algo = DQN(policy=policy,
               qf=qf,
               env_spec=env.spec,
               experiment_id=get_info(),
               plot_list=plots,
               visualizer=visualizer,
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
               auxiliary_objectives=aux_objectives,
               loss_weights=loss_weights)

    # Use modded off policy sampler for passing generating summary statistics about episode's qvals in algo-object.
    runner.setup(algo=algo, env=env, sampler_cls=OffPolicyVectorizedSampler)
    runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)

    # Bypass GarageEnv>>close as this requires a display
    env.env.close()

@ex.main
def run(snapshot_config, exp_config):

    run_task(snapshot_config, exp_config)
