import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    x = None
    if method == "zeros":
        x = torch.zeros(shape) 
    elif method == "ones":
        x = torch.ones(shape)
    elif method == "full":
        x = torch.full(shape, value)
    else:
        raise Exception(f"invalid method {method}")
    return x.tolist()