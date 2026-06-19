import torch.nn as nn
from dl.util import dapply


def adaptively_expand(fn):
    """Decorator to adaptively expand input if needed using exception handling."""
    def wrapper(self, input, *args, **kwargs):
        try:
            # First try the function as-is (assume input is already in correct form)
            return fn(self, input, *args, **kwargs)
        except Exception:
            # If it fails, assume input is a batch of single images and expand it
            input_expanded = input.unsqueeze(1).expand(-1, 2, *input.shape[1:])
            return fn(self, input_expanded, *args, **kwargs)[:, 0, ...]
    return wrapper


class TraceWrapper:
    def zdim(self):
        return int(self.parameters['prop_dim'])

    @adaptively_expand
    def encode(self, input, **kwargs):
        return dapply(dapply(input, self.feature_encoder), self.symbol_net)

    @adaptively_expand
    def decode(self, input, **kwargs):
        return dapply(dapply(input, self.feature_composer),self.feature_decoder)

    @adaptively_expand
    def autoencode(self, input, **kwargs):
        return self.decode(self.encode(input))

    def _build_around(self,input_shape):
        hidden_dim = self.parameters["hidden_dim"]
        self.symbol_net = nn.Sequential(
            nn.Linear(self.fdim(), hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.zdim()),
        )
        self.feature_composer = nn.Sequential(
            nn.Linear(self.zdim(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, self.fdim()),
        )
        super()._build_around(input_shape[1:])

    def _build_aux_around(self,input_shape):
        super()._build_aux_around(input_shape[1:])

    def _build_aux_primary(self, input_shape):
        super()._build_aux_primary(input_shape[1:])