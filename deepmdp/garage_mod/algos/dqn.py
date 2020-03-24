import torch
import numpy as np

from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm

class DQN(OffPolicyRLAlgorithm):

    def __init__(self,
                 env_spec,
                 policy,
                 qf,
                 replay_buffer,
                 exploration_strategy=None,
                 n_epoch_cycles=20,
                 min_buffer_size=int(1e4),
                 buffer_batch_size=64,
                 rollout_batch_size=1,
                 n_train_steps=50,
                 max_path_length=None,
                 qf_lr=0.001,
                 qf_optimizer=torch.optim.Adam,
                 discount=1.0,
                 target_network_update_freq=5,
                 grad_norm_clipping=None,
                 reward_scale=1.,
                 input_include_goal=False,
                 smooth_return=True,
                 name='DQN'
                 ):
        self.qf_lr = qf_lr
        self.qf_optimizer = qf_optimizer
        self.name = name
        self.target_network_update_freq = target_network_update_freq
        self.grad_norm_clipping = grad_norm_clipping

        super(DQN, self).__init__(env_spec=env_spec,
                                  policy=policy,
                                  qf=qf,
                                  exploration_strategy=exploration_strategy,
                                  min_buffer_size=min_buffer_size,
                                  n_train_steps=n_train_steps,
                                  n_epoch_cycles=n_epoch_cycles,
                                  buffer_batch_size=buffer_batch_size,
                                  rollout_batch_size=rollout_batch_size,
                                  replay_buffer=replay_buffer,
                                  max_path_length=max_path_length,
                                  discount=discount,
                                  reward_scale=reward_scale,
                                  input_include_goal=input_include_goal,
                                  smooth_return=smooth_return)

        def optimize_policy(self, itr, samples_data):
            """Optimize policy network."""
            raise NotImplementedError

        def train_once(self, itr, paths):
            """Perform one step of policy optimization given one batch of samples.

            Args:
                itr (int): Iteration number.
                paths (list[dict]): A list of collected paths.

            """
            raise NotImplementedError
