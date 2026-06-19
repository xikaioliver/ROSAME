import numpy as np
import torch
import matplotlib.pyplot as plt
import math


def plot_grid(images, w=10, path="plan.png", verbose=False):
    """
    Arranges and saves a grid of images.

    Parameters:
    - images: List or tensor of images to arrange in a grid.
    - w: Number of images per row (width of the grid).
    - path: File path to save the image grid.
    - verbose: If True, print the path where the image is saved.
    """
    l = len(images)  # Number of images
    h = int(math.ceil(l / w))  # Calculate the number of rows needed
    plt.figure(figsize=(w * 1.5, h * 1.5))

    for i, image in enumerate(images):
        ax = plt.subplot(h, w, i + 1)

        # Check the number of dimensions in the image
        if isinstance(image, torch.Tensor):
            if image.dim() == 2:  # (H, W)
                image = image.unsqueeze(0)  # Add a channel dimension (1, H, W)
            image = image.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) for imshow

        # Plot the image
        plt.imshow(image, interpolation='nearest', cmap='gray', vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if verbose:
        print(path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def squarify(x, dim=-1):
    """
    Reshape the specified dimension of the input tensor into a square matrix.

    Parameters:
    - x: The input tensor.
    - dim: The dimension to be reshaped into a square matrix.

    Returns:
    - The reshaped tensor.
    """
    before, N, after = x.shape[:dim], x.shape[dim], x.shape[dim + 1:]
    if dim == -1:
        after = tuple()

    l = math.ceil(math.sqrt(N))

    if l * l == N:
        return x.reshape((*before, l, l, *after))
    else:
        size = l * l
        padding_size = size - N
        padding_shape = (*before, padding_size, *after)

        # Create a padding tensor of ones with the required shape and same dtype as x
        padding = torch.ones(padding_shape, dtype=x.dtype, device=x.device)

        # Concatenate the original tensor with the padding tensor along the specified dim
        x_padded = torch.cat((x, padding), dim=dim)

        return x_padded.reshape((*before, l, l, *after))


