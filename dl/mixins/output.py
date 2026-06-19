import torch
import torch.nn as nn
import math

class VanillaRenderer:
    def normalize(self, x):
        # Ensure mean and std tensors are on the same device as x
        device = x.device
        mean = torch.tensor(self.parameters["mean"], device=device)
        std = torch.tensor(self.parameters["std"], device=device)
        return (x - mean) / (std + 1e-20)

    def unnormalize(self, x):
        # Ensure mean and std tensors are on the same device as x
        device = x.device
        mean = torch.tensor(self.parameters["mean"], device=device)
        std = torch.tensor(self.parameters["std"], device=device)
        return (x * std) + mean

    def render(self, x, *args, **kwargs):
        return self.unnormalize(x)


class GaussianOutput(VanillaRenderer):
    def __init__(self, sigma=0.1, eval_sigma=None):
        self._sigma = sigma
        self.training = False
        if eval_sigma is None:
            self._eval_sigma = sigma
        else:
            self._eval_sigma = eval_sigma

    def sigma(self):
        return self._sigma if self.training else self._eval_sigma

    def loss(self, true, pred, include_sigma=False):
        sigma = self.sigma()
        v1 = 2 * sigma * sigma
        v2 = 2 * math.pi * sigma * sigma

        loss = (true - pred).pow(2) / v1
        if include_sigma:
            # Ensure the tensor created with v2 is on the same device as `true`
            device = true.device
            loss += torch.log(torch.tensor(v2, device=device)) / 2
        # sum across dimensions
        loss = loss.view(loss.size(0), -1)
        loss = loss.sum(dim=-1)
        return loss

    def activation(self):
        return nn.Identity()
