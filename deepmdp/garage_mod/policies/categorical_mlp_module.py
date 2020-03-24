import torch

from torch.distributions import Categorical
from torch import nn
from garage.torch.modules import MLPModule

class CategoricalMLPModule(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(64, 64),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        # Set output nonlinearity to none as we need raw preds for St gumbel-softmax estimator
        self._output_nonlinearity = None

        self.categorical_logits_module = MLPModule(
            input_dim= self._input_dim,
            output_dim = self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=None,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization
        )

    def forward(self, inputs):
        logits = self.categorical_logits_module(inputs)
        return Categorical(logits=logits)
