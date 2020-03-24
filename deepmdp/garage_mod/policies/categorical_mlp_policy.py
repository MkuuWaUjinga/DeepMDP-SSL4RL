import torch
import akro

from garage.torch.policies import Policy
from deepmdp.garage_mod.policies.categorical_mlp_module import CategoricalMLPModule

class CategoricalMLPPolicy(Policy, CategoricalMLPModule):

    def __init__(self, env_spec, **kwargs):
        assert isinstance(env_spec.action_space, akro.Discrete)
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        Policy.__init__(self, env_spec)
        CategoricalMLPModule.__init__(self,
                                       input_dim=self._obs_dim,
                                       output_dim=self._action_dim,
                                       **kwargs)

    def forward(self, inputs):
        """Forward method."""
        return super().forward(torch.Tensor(inputs))

    def get_action(self, observation):
        """Get a single action given an observation."""
        pass

    def get_actions(self, observations):
        """Get actions given observations."""
        with torch.no_grad():
            dist = self.forward(observations)
            action = dist.sample().view(-1, 1)
            return action.numpy(), dict(mean=dist.mean.numpy(), log_std=(dist.variance**.5).log().numpy()) # why is the variance normalized like this?

    def log_likelihood(self, observation, action):
        """Get log likelihood given observations and action."""
        dist = self.forward(observation)
        return torch.diag(dist.log_prob(action))

    # Todo implement to have entropy-based losses as optimization objective (enforces the policy to act as random as possible)
    def get_entropy(self, observation):
        """Get entropy given observations."""
        return torch.Tensor([])

    def reset(self, dones=None):
        """Reset the environment."""
        pass

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True



