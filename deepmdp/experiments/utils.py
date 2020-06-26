import numpy as np
from visdom import Visdom
from collections import defaultdict
import torch
import pprint

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Visualizer:

    # TODO plot episode lengths
    # TODO plot distribution over chosen actions.

    def __init__(self, experiment_id, plot_list, port=9098):
        self.port = 9098
        self.plot_list = plot_list
        self.viz = Visdom(port=port)
        self.env = experiment_id
        self.line_plotter = VisdomLinePlotter(self.viz, env_name=experiment_id)
        self.correlation_plot_window = None
        self.aux_losses = defaultdict(list)
        self.correlation_matrix = None
        self.num_calls = 0
        self.store_every_th = 10
        self.count_correlation_matrix = 0 # Can be calculated from num_calls and store_every_th

    def publish_config(self, config):
        config_string = pprint.pformat(dict(config)).replace("\n", "<br>").replace(" ", "&nbsp;")
        self.viz.text(config_string, env=self.env)

    def visualize_episodical_stats(self, algo, num_new_episodes):
        if self.make_weights_plot():
            self.visualize_weights(algo, num_new_episodes)
        if self.visualize_aux():
            self.visualize_aux_losses(num_new_episodes, len(algo.episode_rewards))
        if self.visualize_latent_space():
            self.visualize_latent_space_correlation(num_new_episodes, len(algo.episode_rewards), algo.experiment_id)
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

    def visualize_module(self, head, head_name, num_episodes, num_new_episodes):
        for x, params in enumerate(head.parameters()):
            l2_norm = params.data.norm(p=2).cpu().numpy()
            min = torch.min(params.data).cpu().numpy()
            max = torch.max(params.data).cpu().numpy()
            mean = torch.mean(params.data).cpu().numpy()
            for i in range(num_new_episodes):
                self.line_plotter.plot(f"metrics {head_name} {x}", f"L2-norm", f"Weights {head_name} {list(params.shape)}",
                                       num_episodes - num_new_episodes + i, l2_norm)
                self.line_plotter.plot(f"metrics {head_name} {x}", f"min", f"Weights of {head_name} {list(params.shape)}",
                                       num_episodes - num_new_episodes + i, min, color=np.array([[0, 0, 128], ]))
                self.line_plotter.plot(f"metrics {head_name} {x}", f"max", f"Weights of {head_name} {list(params.shape)}",
                                       num_episodes - num_new_episodes + i, max, color=np.array([[128, 0, 0], ]))
                self.line_plotter.plot(f"metrics {head_name} {x}", f"mean", f"Weights of {head_name} {list(params.shape)}",
                                       num_episodes - num_new_episodes + i, mean, color=np.array([[0, 128, 0], ]))


    def visualize_weights(self, algo, num_new_episodes):
        num_episodes = len(algo.episode_rewards)
        self.line_plotter.env = algo.experiment_id + "_weights"
        self.visualize_module(algo.qf.head, "Q-head", num_episodes, num_new_episodes)
        self.visualize_module(algo.qf.encoder, "Encoder", num_episodes, num_new_episodes)
        for aux in algo.auxiliary_objectives:
            self.visualize_module(aux.net, aux.__class__.__name__, num_episodes, num_new_episodes)
        self.line_plotter.env = algo.experiment_id + "_main"


    def make_weights_plot(self):
        return "weight_plot" in self.plot_list

    def visualize_aux(self):
        return "aux_loss_plot" in self.plot_list

    def visualize_latent_space(self):
        return "latent_space_correlation_plot" in self.plot_list

    def visualize_stats(self):
        return "episodical_stats" in self.plot_list

    def save_aux_loss(self, loss, loss_type):
        if self.visualize_aux():
            self.aux_losses[loss_type].append(loss)

    def visualize_aux_losses(self, num_new_episodes, total_num_episode):
        if self.aux_losses and num_new_episodes > 0:
            for aux_loss in self.aux_losses:
                for i in range(num_new_episodes):
                    self.line_plotter.plot(aux_loss, "mean", aux_loss, total_num_episode - num_new_episodes + i,
                                   np.mean(self.aux_losses[aux_loss]))
                    self.line_plotter.plot(aux_loss, "median", aux_loss, total_num_episode - num_new_episodes + i,
                                   np.median(self.aux_losses[aux_loss]), color=np.array([[0, 0, 128], ]))
            self.aux_losses = defaultdict(list)

    def save_latent_space(self, algo, next_obs, ground_truth_embedding):
        if self.visualize_latent_space() and self.num_calls % self.store_every_th == 0:
            if ground_truth_embedding is None:
                raise ValueError("Ground truth embedding mustn't be of None type")
            ground_truth_embedding = ground_truth_embedding.to(device)
            algo.qf.eval()
            with torch.no_grad():
                _, embedding = algo.qf(next_obs, return_embedding=True)
            algo.qf.train()
            assert embedding.size() == ground_truth_embedding.size()
            if self.correlation_matrix is None:
                embedding_dim = embedding.size(1)
                self.correlation_matrix = torch.zeros((embedding_dim, embedding_dim)).to(device)
            # Calculate correlation
            self.correlation_matrix += self.calculate_correlation(embedding.t(), ground_truth_embedding.t())
            self.count_correlation_matrix += 1
        self.num_calls += 1

    def visualize_latent_space_correlation(self, num_new_episodes, total_num_episodes, experiment_id):
        if self.correlation_matrix is not None and num_new_episodes > 0:
            self.correlation_matrix = self.correlation_matrix.div(self.count_correlation_matrix)
            assert round(torch.max(self.correlation_matrix).item(), 2) <= 1.0 and round(torch.min(
                self.correlation_matrix).item(), 2) >= -1.0, "Invalid value for correlation coefficient!"
            self.line_plotter.env = experiment_id + "_latent_space"
            column_names = ["pos_x", "pos_y", "vel_x", "vel_y", "ang", "ang_vel", "leg_1", "leg_2"]
            row_names = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8']
            self.correlation_plot_window = self.viz.heatmap(X=self.correlation_matrix,
                                                            env=self.env,
                                                            win=self.correlation_plot_window,
                                                            opts=dict(
                                                                columnnames=column_names,
                                                                rownames= row_names,
                                                                colormap='Viridis',
                                                                xmin=-1.0,
                                                                xmax=1.0,
                                                                title="Average latent space correlation per batch and episode"
                                                            ))
            for i, column_name in enumerate(column_names):
                for j, row_name in enumerate(row_names):
                    for k in range(num_new_episodes):
                        self.line_plotter.plot(column_name + "_correlation", row_name, column_name,
                                               total_num_episodes - num_new_episodes + k,
                                               self.correlation_matrix[j, i].cpu().numpy(),
                                               color=np.array([[int((255/8)*j), int((255/8)*(8-j)), 0],]))
            self.line_plotter.env = experiment_id + "_main"
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

            assert round(torch.max(c).item(), 2) <= 1.0 and round(torch.min(
                c).item(), 2) >= -1.0, "Invalid value for correlation coefficient!"
            return c


class VisdomLinePlotter:

    def __init__(self, viz, env_name='main', xlabel='episode number'):
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
