import torch
import numpy as np

from dowel import tabular, logger
from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.misc.tensor_utils import normalize_pixel_batch
from deepmdp.experiments.utils import VisdomLinePlotter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(OffPolicyRLAlgorithm):

    def __init__(self,
                 env_spec,
                 policy,
                 qf,
                 replay_buffer,
                 exploration_strategy=None,
                 n_epoch_cycles=20,
                 min_buffer_size=int(1e4),
                 buffer_batch_size=32,
                 rollout_batch_size=1,
                 n_train_steps=50,
                 max_path_length=None,
                 qf_lr=0.0002,
                 qf_optimizer=torch.optim.Adam,
                 discount=0.99,
                 reward_scale=1.,
                 target_network_update_freq=5,
                 input_include_goal=False,
                 smooth_return=True,
                 ):
        super(DQN, self).__init__(env_spec=env_spec,
                                  policy=policy,
                                  qf=qf ,
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
        self.qf_lr = qf_lr
        self.qf_optimizer = qf_optimizer(qf.parameters(), lr=qf_lr)
        self.target_network_update_freq = target_network_update_freq
        self.episode_rewards = []
        self.episode_mean_q_vals = []
        self.episode_qf_losses = []
        self.episode_std_q_vals = []
        # Clone target q-network
        self.target_qf = self.qf.clone()
        self.target_qf.to(device)
        self.qf.to(device)
        self.visdom = VisdomLinePlotter(9098, xlabel="episode number")
        logger.log(f"Number of parameter of q-network are: {sum(p.numel() for p in qf.parameters() if p.requires_grad)}")


    def optimize_policy(self, itr, samples):
        """Optimize policy network."""
        del itr
        del samples
        action_dim = self.env_spec.action_space.n
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        observations = transitions['observation']
        rewards = torch.FloatTensor(transitions['reward'])
        actions = torch.FloatTensor(transitions['action'])
        next_observations = transitions['next_observation']
        dones = torch.FloatTensor(transitions['terminal'])

        # Obs. are stored in uint8 format in replay buffer to optimize memory.
        # Convert pixel values to [0,1] for training if env's obs are images.
        observations = normalize_pixel_batch(self.env_spec, observations)
        next_observations = normalize_pixel_batch(self.env_spec, next_observations)
        with torch.no_grad():
            target_qvals = self.target_qf(next_observations)
            target_qvals, _ =  torch.max(target_qvals, dim=1)
            assert target_qvals.size(0) == self.buffer_batch_size, "number of target qvals has to equal batch size"

        # if done, it's just reward else reward + discount * future_best_q_val
        target = rewards + (1.0 - dones) * self.discount * target_qvals

        qval = self.qf(observations)
        actions = self.one_hot(actions, action_dim) # Todo is there a better way to do this?
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

        # wether the agent's estimation of value was correct.
        self.episode_rewards.extend(paths['undiscounted_returns'])
        self.episode_mean_q_vals.extend(paths['episode_mean_q_vals'])
        self.episode_std_q_vals.extend(paths['episode_std_q_vals'])
        for i in range(len(self.episode_rewards) - len(paths["undiscounted_returns"]), len(self.episode_rewards)):
            self.visdom.plot("episode reward", "rewards", "Rewards per episode", i, self.episode_rewards[i])
            self.visdom.plot("episode mean q-values", "q-values", "Mean q-values per episode", i, self.episode_mean_q_vals[i])
            self.visdom.plot("episode std q-values", "q-std", "Std of q-values per episode", i, self.episode_std_q_vals[i])

        last_average_return = np.mean(self.episode_rewards)
        for _ in range(self.n_train_steps):
            if self._buffer_prefilled:
                qf_loss = self.optimize_policy(itr, None)
                self.episode_qf_losses.append(qf_loss)

        if self._buffer_prefilled:
            if itr % self.target_network_update_freq == 0:
                self.update_target()

        if itr % self.n_epoch_cycles == 0:
            if self._buffer_prefilled:
                mean100ep_rewards = round(np.mean(self.episode_rewards[-100:]), 1)
                mean100ep_q_vals = round(np.mean(self.episode_mean_q_vals[-100:]), 1)
                mean100ep_qf_loss = np.mean(self.episode_qf_losses[-100:])
                tabular.record('Epoch', epoch)
                tabular.record("Episode100QValuesMean", mean100ep_q_vals)
                tabular.record('Episode100RewardMean', mean100ep_rewards)
                tabular.record('Episode100LossMean', mean100ep_qf_loss)
        return last_average_return

    def process_samples(self, itr, paths):
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            dict: Processed sample data, with keys
                * undiscounted_returns (list[float])
                * success_history (list[float])
                * complete (list[bool])

        """
        success_history = [
            path['success_count'] / path['running_length'] for path in paths
        ]
        undiscounted_returns = [path['undiscounted_return'] for path in paths]
        episode_mean_q_vals = [np.mean(path['q_vals']) for path in paths]
        episode_std_q_vals = [np.std(path["q_vals"]) for path in paths]

        # check if the last path is complete
        complete = [path['dones'][-1] for path in paths]

        samples_data = dict(undiscounted_returns=undiscounted_returns,
                            episode_mean_q_vals=episode_mean_q_vals,
                            episode_std_q_vals=episode_std_q_vals,
                            success_history=success_history,
                            complete=complete)

        return samples_data

    def update_target(self, tau:int=1):
        """
        Update target network with q-network's parameters.
        :param tau: Fraction to update. Default is hard update.
        """
        logger.log("Updating target network")
        for t_param, param in zip(self.target_qf.parameters(),
                                  self.qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - tau) +
                               param.data * tau)

    @property
    def _buffer_prefilled(self):
        """Flag whether first min buffer size steps are done."""
        return self.replay_buffer.n_transitions_stored >= self.min_buffer_size

    def __getstate__(self):
        """Return state values to be pickled."""
        data = self.__dict__.copy()
        del data['visdom']
        return data

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__ = state
        self.visdom = VisdomLinePlotter(9098, xlabel="episode number")