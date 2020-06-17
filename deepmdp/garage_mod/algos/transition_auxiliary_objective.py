import torch
from dowel import logger
from deepmdp.garage_mod.algos.auxiliary_objective import AuxiliaryObjective
from garage.torch.modules.mlp_module import MLPModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransitionAuxiliaryObjective(AuxiliaryObjective):

    def __init__(self,
                 env_spec,
                 input_dim,
                 head_config,
                 nonlinearity=torch.nn.ReLU,
                 w_init=torch.nn.init.xavier_normal_,
                 b_init=torch.nn.init.zeros_):

        self.head_config = head_config
        self._env_spec = env_spec
        self.input_dim = input_dim
        self.action_dim = self._env_spec.action_space.flat_dim
        if "filter_dims" in self.head_config:
            conv = torch.nn.Conv2d(
                    in_channels=self.input_dim,
                    out_channels=self.input_dim * self.action_dim,
                    kernel_size=2,
                    stride=1,
                ).to(device)
            w_init(conv.weight)
            b_init(conv.bias)
            zero_pad = torch.nn.ZeroPad2d((0, 1, 0, 1)) # maintain input dimensionality.
            self.net = torch.nn.Sequential(zero_pad, conv, nonlinearity())
        elif "dense_sizes" in self.head_config:
            self.net = MLPModule(input_dim=self.input_dim,
                                 output_dim=self.input_dim * self.action_dim,
                                 hidden_sizes=self.head_config["dense_sizes"],
                                 hidden_nonlinearity=torch.nn.ReLU,
                                 hidden_w_init=w_init,
                                 hidden_b_init=b_init,
                                 output_nonlinearity=None,
                                 output_w_init=w_init,
                                 output_b_init=b_init)
        self.net.to(device)

        logger.log(f"Transition net: {self.net}")

    def compute_loss(self, embedding, embedding_next_observation, actions):
        """
        Compute loss between embedding of next observation and the predicted embedding of the next observation.
        :param embedding: The embedded observation used as an input for the latent transition network
        :param embedding_next_observation: The ground truth embedded next obervation
        :param actions: The actions that caused the embedded next_observations.
        :return: The mean squared error between the predicted and the ground truth embedding of the next observation.
        """
        preds = self.net(embedding)
        batch_size = actions.size(0)
        # Reshape tensor: B x act * channels ... --> B x channels x ... x act
        preds = preds.unsqueeze(len(preds.size())).reshape(batch_size, self.input_dim, *preds.size()[2:4], self.action_dim)
        loss_func = torch.nn.MSELoss()
        loss = 0
        for i, act in enumerate(actions):
            predicted__next_observation_embedding = preds[i, ..., int(act.item())].squeeze()
            ground_truth_embedding = embedding_next_observation[i, ...]
            assert(ground_truth_embedding.size() == predicted__next_observation_embedding.size())
            loss += loss_func(predicted__next_observation_embedding, ground_truth_embedding)
        return loss