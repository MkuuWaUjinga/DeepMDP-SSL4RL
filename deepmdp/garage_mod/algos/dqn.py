import torch
import numpy as np

from dowel import tabular
from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.torch.utils import np_to_torch, torch_to_np

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
                 reward_scale=1.,
                 target_network_update_freq=5,
                 input_include_goal=False,
                 smooth_return=True,
                 name='DQN'
                 ):
        self.qf_lr = qf_lr
        self.qf_optimizer = qf_optimizer(qf.parameters(), lr=qf_lr)
        self.name = name
        self.target_network_update_freq = target_network_update_freq
        self.episode_rewards = []
        self.episode_qf_losses = []

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


    def optimize_policy(self):
        """Optimize policy network."""
        action_dim = self.env_spec.action_space.n
        transitions = np_to_torch(self.replay_buffer.sample(self.buffer_batch_size))
        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        dones = transitions['terminal']

        with torch.no_grad():
            target_qvals = torch.max(self.qf(next_observations))

        # if done, it's just reward
        # else reward + discount * future_best_q_val
        target = rewards + (1.0 - dones) * self.discount * target_qvals

        qval = self.qf(observations)
        actions = self.one_hot(actions, action_dim)
        q_selected = torch.sum(qval*actions, axis=1)

        qf_loss = torch.nn.MSELoss()
        qval_loss = qf_loss(q_selected, target)
        self.qf_optimizer.zero_grad()
        qval_loss.backward()
        self.qf_optimizer.step()
        return qval_loss.detach()

    def one_hot(self, action, action_dim):
        y_onehot = torch.FloatTensor(self.buffer_batch_size, action_dim)
        y_onehot.zero_()
        y_onehot.scatter_(1, action.long().view(-1, 1), 1)
        return y_onehot

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """
        paths = self.process_samples(itr, paths)
        epoch = itr / self.n_epoch_cycles

        self.episode_rewards.extend(paths['undiscounted_returns'])
        last_average_return = np.mean(self.episode_rewards)
        for _ in range(self.n_train_steps):
            if self._buffer_prefilled:
                qf_loss = self.optimize_policy()
                self.episode_qf_losses.append(qf_loss)

        if self._buffer_prefilled:
            if itr % self.target_network_update_freq == 0:
                self.update_target()

        if itr % self.n_epoch_cycles == 0:
            if self._buffer_prefilled:
                mean100ep_rewards = round(np.mean(self.episode_rewards[-100:]),
                                          1)
                mean100ep_qf_loss = np.mean(self.episode_qf_losses[-100:])
                tabular.record('Epoch', epoch)
                tabular.record('Episode100RewardMean', mean100ep_rewards)
                tabular.record('Episode100LossMean', mean100ep_qf_loss)
        return last_average_return

    def update_target(self):
        pass

    @property
    def _buffer_prefilled(self):
        """bool: Whether first min buffer size steps is done."""
        return self.replay_buffer.n_transitions_stored >= self.min_buffer_size