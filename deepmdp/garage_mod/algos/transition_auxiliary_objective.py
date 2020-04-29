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

        action_dim = self._env_spec.action_space.flat_dim
        conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * action_dim,
                kernel_size=2,
                stride=1,
            ).to(device)
        w_init(conv.weight)
        b_init(conv.bias)
        zero_pad = torch.nn.ZeroPad2d((0, 1, 0, 1)) # maintain input dimensionality.
        self.latent_transition_network = torch.nn.Sequential(zero_pad, conv, nonlinearity())

    def compute_loss(self, embedding, transition, actions):
        preds = self.latent_transition_network(embedding)
        mask = np.zeros((len(actions), preds.size()[1], preds.size()[2], preds.size()[3]))
        for i, act in enumerate(actions):
            mask[i, 64*int(act.item()):64*(int(act.item())+1), :, :] = 1
        mask = torch.from_numpy(mask).bool()
        predicted_transition_selected_by_action = torch.masked_select(preds, mask)
        loss_func = torch.nn.MSELoss()
        return loss_func(predicted_transition_selected_by_action, transition.flatten())