import torch

def initialize_weights(fan_in, fan_out, method):
    weights = torch.empty(fan_out, fan_in)
    if method == "xavier_uniform":
        a = (6.0 / (fan_in + fan_out)) ** 0.5
        weights.uniform_(-a, a)
    elif method == "xavier_normal":
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        weights.normal_(0.0, std)
    elif method == "he_uniform":
        a = (6.0 / fan_in) ** 0.5
        weights.uniform_(-a, a)
    elif method == "he_normal":
        std = (2.0 / fan_in) ** 0.5
        weights.normal_(0.0, std)
    else:
        raise ValueError("Unknown initialization method")

    return weights
