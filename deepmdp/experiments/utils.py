import numpy as np
from visdom import Visdom
from collections import defaultdict
import torch


def show_agent_playing(policy, env):
    obs = env.reset()
    env.render()
    for i in range(1000):
        action = policy.get_actions(obs)
        print(action[0])
        obs, rewards, done, env_infos = env.step(action[0])
        if done:
            break
        env.render()


class Visualizer:

    def __init__(self, experiment_id, plot_list, port=9098):
        self.port = 9098
        self.plot_list = plot_list
        self.viz = Visdom(port=port)
        self.env = experiment_id
        self.line_plotter = VisdomLinePlotter(self.viz, env_name=experiment_id)
        self.correlation_plot_window = None
        self.aux_losses = defaultdict(list)
        self.correlation_matrix = None
        self.count_correlation_matrix = 0

    def visualize_episodical_stats(self, algo, num_new_episodes):
        if self.visualize_stats():
            for i in range(len(algo.episode_rewards) - num_new_episodes, len(algo.episode_rewards)):
                self.line_plotter.plot("episode reward", "rewards", "Rewards per episode", i, algo.episode_rewards[i])
                self.line_plotter.plot("episode mean q-values", "q-values", "Mean q-values per episode", i,
                                       algo.episode_mean_q_vals[i])
                self.line_plotter.plot("episode std q-values", "q-std", "Std of q-values per episode", i,
                                       algo.episode_std_q_vals[i])
                # Plot running average of rewards
                if i > 100:
                    self.line_plotter.plot("episode reward", "avg reward", "Rewards per episode", i,
                                           np.mean(algo.episode_rewards[i - 100:i]),
                                           color=np.array([[0, 0, 128], ]))

    def visualize_aux(self):
        return "aux_loss_plot" in self.plot_list

    def visualize_latent_space(self):
        return "latent_space_correlation_plot" in self.plot_list

    def visualize_stats(self):
        return "episodical_stats" in self.plot_list

    def save_aux_loss(self, loss, loss_type):
        if self.visualize_aux():
            self.aux_losses[loss_type].append(loss)

    def visualize_aux_losses(self, iteration):
        if self.aux_losses:
            self.line_plotter.xlabel = "training iterations"
            for aux_loss in self.aux_losses:
                self.line_plotter.plot(aux_loss, aux_loss, aux_loss, iteration, np.mean(self.aux_losses[aux_loss]))
            self.aux_losses = defaultdict(list)

    def save_latent_space(self, algo, next_obs, ground_truth_embedding):
        if self.visualize_latent_space():
            if ground_truth_embedding is None:
                raise ValueError("Ground truth embedding mustn't be of None type")
            algo.qf.eval()
            with torch.no_grad():
                _, embedding = algo.qf(next_obs, return_embedding=True)
            algo.qf.train()
            assert embedding.size() == ground_truth_embedding.size()
            if self.correlation_matrix is None:
                embedding_dim = embedding.size(1)
                self.correlation_matrix = torch.zeros((embedding_dim, embedding_dim))
            # Calculate correlation
            self.correlation_matrix += self.calculate_correlation(embedding.t(), ground_truth_embedding.t())
            self.count_correlation_matrix += 1

    def visualize_training_results(self, itr):
        if self.visualize_aux():
            self.visualize_aux_losses(itr)
        if self.visualize_latent_space():
            self.visualize_latent_space_correlation()

    def visualize_latent_space_correlation(self):
        if self.correlation_matrix is not None:
            self.correlation_matrix = self.correlation_matrix.div(self.count_correlation_matrix)
            assert torch.max(self.correlation_matrix).item() <= 1.0 and torch.min(
                self.correlation_matrix).item() >= -1.0, "Invalid value for correlation coefficient!"
            self.correlation_plot_window = self.viz.heatmap(X=torch.abs(self.correlation_matrix),
                                                            env=self.env,
                                                            win=self.correlation_plot_window,
                                                            opts=dict(
                                                                columnnames=["pos_x", "pos_y", "vel_x", "vel_y", "ang",
                                                                             "ang_vel", "leg_1", "leg_2"],
                                                                rownames=['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7',
                                                                          'l8'],
                                                                colormap='Viridis',
                                                                xmin=0,
                                                                xmax=1.0,
                                                                title="Latent space correlation with ground truth state"
                                                            ))
            self.correlation_matrix = None
            self.count_correlation_matrix = 0

    @staticmethod
    def calculate_correlation(x1, x2):
        """
        takes two 2D tensors of (latent_space_size, sample_size) and calculates the column-wise correlation between the
        two
        :param x1:
        :param x2:
        :return: a 2D tensor of shape (latent_space_size, latent_space_size)
        """
        with torch.no_grad():
            # Calculate covariance matrix of columns
            mean_x1 = torch.mean(x1, 1).unsqueeze(1)
            mean_x2 = torch.mean(x2, 1).unsqueeze(1)
            x1m = x1.sub(mean_x1)
            x2m = x2.sub(mean_x2)
            c = x1m.mm(x2m.t())
            c = c / (x1.size(1) - 1)
            # Normalize by standard deviations. Add epsilon for numerical stability if std close to 0
            epsilon = 1e-9
            std_x1 = torch.std(x1, 1).unsqueeze(1) + epsilon
            std_x2 = torch.std(x2, 1).unsqueeze(1) + epsilon
            c = c.div(std_x1)
            c = c.div(std_x2.t())

            assert torch.max(c).item() <= 1.0 and torch.min(
                c).item() >= -1.0, "Invalid value for correlation coefficient!"
            return c


class VisdomLinePlotter:

    def __init__(self, viz, env_name='main', xlabel='Iteration'):
        self.viz = viz
        self.env = env_name
        self.plots = {}
        self.legend = defaultdict(dict)
        self.xlabel = xlabel

    def plot(self, var_name, split_name, title_name, x, y, color=np.array([[255, 136, 0], ])):
        self.legend[var_name][split_name] = None
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                title=title_name,
                xlabel=self.xlabel,
                ylabel=var_name,
                linecolor=color,
                legend=list(self.legend[var_name])
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append', opts=dict(linecolor=color, legend=list(self.legend[var_name])))
