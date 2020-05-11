# Neural Network variante
# Minimiere Wasserstein distanz zwischen DeepMDP.

import torch
import numpy as np

from deepmdp.garage_mod.algos.auxiliary_objective import AuxiliaryObjective
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransitionAuxiliaryObjective(AuxiliaryObjective):

    def __init__(self,
                 env_spec,
                 in_channels,
                 nonlinearity=torch.nn.ReLU,
                 w_init=torch.nn.init.xavier_normal_,
                 b_init=torch.nn.init.zeros_):

        self._env_spec = env_spec
        self.in_channels = in_channels
        self.action_dim = self._env_spec.action_space.flat_dim

        conv = torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels * self.action_dim,
                kernel_size=2,
                stride=1,
            ).to(device)
        w_init(conv.weight)
        b_init(conv.bias)
        zero_pad = torch.nn.ZeroPad2d((0, 1, 0, 1)) # maintain input dimensionality.
        self.latent_transition_network = torch.nn.Sequential(zero_pad, conv, nonlinearity())

    def compute_loss(self, embedding, embedding_next_observation, actions):
        """
        Compute loss between embedding of next observation and the predicted embedding of the next observation.
        :param embedding: The embedded observation used as an input for the latent transition network
        :param embedding_next_observation: The ground truth embedded next obervation
        :param actions: The actions that caused the embedded next_observations.
        :return: The mean squared error between the predicted and the ground truth embedding of the next observation.
        """
        preds = self.latent_transition_network(embedding)
        batch_size = actions.size(0)
        # Reshape tensor: B x act * channels ... --> B x channels x ... x act
        preds = preds.unsqueeze(len(preds.size())).reshape(batch_size, self.in_channels, *preds.size()[2:4], self.action_dim)
        loss_func = torch.nn.MSELoss()
        loss = 0
        for i, act in enumerate(actions):
            predicted__next_observiation_embedding = preds[i, ..., int(act.item())].squeeze()
            ground_truth_embedding = embedding_next_observation[i, ...]
            assert(ground_truth_embedding.size() == predicted__next_observiation_embedding.size())
            loss += loss_func(predicted__next_observiation_embedding, ground_truth_embedding)
        return loss