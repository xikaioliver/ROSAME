from ..util.layers import *

# "Action" Network
class DetActionMixin:
    "Deterministic mapping from a state pair to an action"
    def build_action_fc_unit(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.parameters["aae_width"]),
            nn.ReLU(),
        )

    def st(self, x):
        # Use straight-through estimator
        argmax = F.one_hot(torch.argmax(x, dim=-1), num_classes=x.size(-1)).float()
        softmax = F.softmax(x, dim=-1)
        return (argmax - softmax).detach() + softmax

    def _build_around(self, input_shape):
        # Build the action encoder network
        layers = [
            self.build_action_fc_unit(2 * self.zdim() if i==0 else self.parameters["aae_width"])
            for i in range(self.parameters["aae_depth"] - 1)
        ]

        self.action = nn.Sequential(
            *layers,
            nn.Linear(self.parameters["aae_width"], self.adim()),
        )
        self.action_activation = nn.Softmax(dim=-1)

        # Call to the parent class's _build_around if needed
        super()._build_around(input_shape)

    def _build_aux_around(self, input_shape):
        super()._build_aux_around(input_shape)
        return

    def adim(self):
        return self.parameters["action_dim"]

    def encode_action(self, z_pre, z_suc):
        return self.action_activation(self.action(torch.cat([z_pre, z_suc], dim=-1)))