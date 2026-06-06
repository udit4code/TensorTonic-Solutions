import torch

def relu(x):
    # We need to clamp down all elements of x which are negative 
    return torch.clamp(x, min=0)

def sigmoid(x):
    # This is not yet numerically stable. We can do many production-grade optimisations.
    return 1.0 / (1.0 + torch.exp(-x))

def tanh(x):
    # This is not yet numerically stable. We can do many production-grade optimisations.
    exp_pos = torch.exp(x)
    exp_neg = torch.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)

def leaky_relu(x):
    return torch.where(x > 0, x, 0.01 * x)
    
def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    result = None
    x = torch.tensor(x, dtype=torch.float32)
    if method == "relu":
        result = relu(x) 
    elif method == "sigmoid":
        result = sigmoid(x) 
    elif method == "tanh":
        result = tanh(x)
    elif method == "leaky_relu":
        result = leaky_relu(x) 
    else:
        raise Exception(f"invalid method : {method}")
    return result.tolist()