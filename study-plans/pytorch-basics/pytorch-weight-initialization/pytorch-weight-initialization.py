import torch
import torch.nn as nn

def initialize_weights(fan_in, fan_out, method):
    """
    Returns:
        Tensor of shape (fan_out, fan_in) initialized using
        the specified method.
    """

    # Create an empty weight tensor.
    weights = torch.empty(fan_out, fan_in)

    if method == "xavier_uniform":
        nn.init.xavier_uniform_(weights)

    elif method == "xavier_normal":
        nn.init.xavier_normal_(weights)

    elif method == "he_uniform":
        nn.init.kaiming_uniform_(weights, mode="fan_in", nonlinearity="relu")

    elif method == "he_normal":
        nn.init.kaiming_normal_(weights, mode="fan_in", nonlinearity="relu")

    else:
        raise ValueError(f"Unknown initialization method: {method}")

    return weights
