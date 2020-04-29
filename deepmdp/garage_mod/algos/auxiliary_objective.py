import torch

class AuxiliaryObjective:

    def compute_loss(self, embedding:torch.Tensor, targets:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        pass
