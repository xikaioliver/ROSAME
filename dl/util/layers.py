import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *shape):
        """
        Reshape layer for PyTorch, similar to Keras' Reshape layer.

        Args:
            shape (tuple): The desired shape of each input tensor (excluding the batch size).
        """
        super(Reshape, self).__init__()
        if len(shape) == 1 and isinstance(shape[0], torch.Size):
            self.shape = tuple(shape[0])  # Convert torch.Size to a tuple
        else:
            self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)  # x.size(0) is the batch size