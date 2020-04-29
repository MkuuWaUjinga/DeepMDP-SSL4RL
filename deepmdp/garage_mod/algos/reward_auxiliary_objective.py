import torch

from garage.torch.modules.mlp_module import MLPModule
from deepmdp.garage_mod.algos.auxiliary_objective import AuxiliaryObjective

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RewardAuxiliaryObjective(AuxiliaryObjective):

    def __init__(self,
                 env_spec,
                 embedding_dim,
                 output_nonlinearity=None,
                 output_w_init=torch.nn.init.xavier_normal_,
                 output_b_init=torch.nn.init.zeros_):

        self._env_spec = env_spec
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init

        action_dim = self._env_spec.action_space.flat_dim

        self.reward_network = MLPModule(
            input_dim=embedding_dim,
            output_dim=action_dim,
            hidden_sizes=[],
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init
        ).to(device) # fully-connected 1 x num_actions outputs


    def compute_loss(self, embedding, rewards, actions):
        assert list(actions.size()) == [32, 4]
        preds = self.reward_network(embedding)
        selected_predicted_rewards = torch.sum(preds * actions, axis = 1)
        loss_func = torch.nn.SmoothL1Loss()
        return loss_func(selected_predicted_rewards, rewards)

