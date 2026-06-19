import os
import torch
from dl.util.plot import squarify, plot_grid


class TracePlotMixin:
    @torch.no_grad()
    def plot_plan(self, z, style=None, path="", verbose=False):
        """Plot a sequence of states horizontally.
        If no style vector is provided, sample the style from standard normal.
        """
        self.net.eval()
        y = self.decode(z)
        y = self.render(y)
        y = [r for r in y]
        plot_grid(y, w=8, path=path, verbose=True)
        return

    @torch.no_grad()
    def plot_transitions(self, x, path, verbose=False, epoch=None):
        self.net.eval()
        x = x.to(next(self.net.parameters()).device)
        basename, ext = os.path.splitext(path)

        # Encode input into symbols and styles
        z = torch.sigmoid(self.encode(x))  # symbols: [B, T, symbol_dim]
        y = self.decode(z)

        # Prepare forward inference
        z_pre, z_suc = z[:, :-1, ...], z[:, 1:, ...]
        actions = self.encode_action(z_pre, z_suc)  # Actions based on adjacent symbols
        z_suc_aae = self.apply(z_pre, actions)  # Forward inference

        # Decode forward inferred symbols into images
        y_suc_aae = self.decode(z_suc_aae)  # [B, T-1, style_dim]

        # Render
        x = self.render(x)
        y = self.render(y)
        y_suc_aae =  self.render(y_suc_aae)

        # Create empty tensors on the same device as the target tensor
        empty_image = torch.ones_like(x[:, 0, ...])
        empty_symbol = torch.ones_like(z[:, 0, ...])

        # Use these tensors in your concatenation
        y_suc_aae = torch.cat((empty_image.unsqueeze(1), y_suc_aae),
                                          dim=1)  # Prepend empty image
        z_suc_aae = torch.cat((empty_symbol.unsqueeze(1), z_suc_aae), dim=1)  # Prepend empty symbol

        B = x.shape[0]
        for i in range(B):
            self._plot(basename + f"_sample_{i}_transition_image".format(i) + ext,
                       [x[i], y[i], y_suc_aae[i]], epoch=epoch, orientation="rows")

            self._plot(basename + f"_sample_{i}_transition_latent".format(i) + ext,
                       map(squarify, [z[i], z_suc_aae[i]]),
                       epoch=epoch, orientation="rows")

        return