import torch
from dowel import logger
from garage.torch.modules.mlp_module import MLPModule
from deepmdp.garage_mod.algos.auxiliary_objective import AuxiliaryObjective

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RewardAuxiliaryObjective(AuxiliaryObjective):

    def __init__(self,
                 env_spec,
                 embedding_dim,
                 head_config,
                 output_nonlinearity=None,
                 output_w_init=torch.nn.init.xavier_normal_,
                 output_b_init=torch.nn.init.zeros_):

        self._env_spec = env_spec
        self._head_config = head_config
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        assert "dense_sizes" in head_config

        action_dim = self._env_spec.action_space.flat_dim

        self.net = MLPModule(
            input_dim=embedding_dim,
            output_dim=action_dim,
            hidden_sizes=self._head_config["dense_sizes"],
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init
        ) # fully-connected 1 x num_actions outputs
        self.net.to(device)

        logger.log(f"Reward net: {self.net}")


    def compute_loss(self, embedding, rewards, actions):
        assert list(actions.size()) == [32, 4]
        preds = self.net(embedding)
        selected_predicted_rewards = torch.sum(preds * actions, axis = 1)
        loss_func = torch.nn.SmoothL1Loss()
        return loss_func(selected_predicted_rewards, rewards)

