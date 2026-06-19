import json
import numpy as np
import torch

def ensure_list(x):
    if type(x) is not list:
        return [x]
    else:
        return x

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert PyTorch tensor to a list
        else:
            return super(NpEncoder, self).default(obj)

def count_params(model):
    # Count the total number of trainable parameters in the model
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_count

def dapply(x, fn=None):
    """
    Take a batch of traces of elements (e.g. shape : [B, T, ...]), apply fn to each element
    ([B, ...]), then concatenate the result into traces ([B, T, result_shape]).
    Input: [B, T, ...]
    Output:
        - Entire result applied across all elements in the trace: [B, T, ...]
    """
    B, T, *other_dims = x.shape

    if fn is not None:
        # Flatten traces for processing
        x_flat = x.reshape(B * T, *other_dims)  # [B * T, ...]
        y_flat = fn(x_flat)  # Apply function across flattened batch
        feature_dim = y_flat.shape[1:]  # Extract feature dimensions after processing
        y = y_flat.view(B, T, *feature_dim)  # Reshape back to [B, T, feature_dim]
    else:
        y = x  # If no function, return input as-is

    return y


def combine_styles(styles, style_combination_mode="average"):
    """
    Combine styles using the specified mode, ensuring the first and last steps
    use the first and last styles, respectively, and intermediate steps average
    their previous and next styles or concatenate them.

    Args:
        styles (torch.Tensor): Tensor of shape [B, T-1, style_dim], representing the styles.
        style_combination_mode (str): Mode to combine the styles, either "average" or "concat".

    Returns:
        torch.Tensor: Combined styles of shape [B, T, style_dim] (for "average")
                     or [B, T, 2*style_dim] (for "concat").

    Raises:
        ValueError: If an unknown style_combination_mode is provided.
    """
    B, T_minus_1, style_dim = styles.shape

    # Create padded styles by replicating the first and last styles
    style_prev = torch.cat([styles[:, 0:1, :], styles], dim=1)  # [B, T, style_dim]
    style_next = torch.cat([styles, styles[:, -1:, :]], dim=1)  # [B, T, style_dim]

    if style_combination_mode == "average":
        # Average of previous and next styles
        combined_styles = (style_prev + style_next) / 2  # [B, T, style_dim]
    elif style_combination_mode == "concat":
        # Concatenate previous and next styles along the last dimension
        combined_styles = torch.cat((style_prev, style_next), dim=-1)  # [B, T, 2*style_dim]
    else:
        raise ValueError(f"Unknown style_combination_mode: {style_combination_mode}")

    return combined_styles


def repeat_styles(styles):
    """
    Generate three variations of styles: combined, front-extended, and back-extended.

    Args:
        styles (torch.Tensor): The styles tensor of shape [B, T-1, style_dim].

    Returns:
        tuple: Three style tensors (combined_styles, front_extended_styles, back_extended_styles).
    """
    # Combine styles using the average mode
    combined_styles = combine_styles(styles, "average")  # [B, T, style_dim]

    # Extend styles by repeating the first and last style
    front_extended_styles = torch.cat((styles[:, :1, :], styles), dim=1)  # [B, T, style_dim]
    back_extended_styles = torch.cat((styles, styles[:, -1:, :]), dim=1)  # [B, T, style_dim]

    return combined_styles, front_extended_styles, back_extended_styles


def gpu_info():
    result = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_properties = torch.cuda.get_device_properties(i)
            gpu_info = {
                "name": gpu_properties.name,
                "capability": f"{gpu_properties.major}.{gpu_properties.minor}",
                "total_memory": f"{gpu_properties.total_memory / (1024**3):.2f} GB",
                "multi_processor_count": gpu_properties.multi_processor_count,
            }
            result.append(gpu_info)
    return result


def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

