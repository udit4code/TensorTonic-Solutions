import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    x = None
    if method == "zeros":
        x = torch.zeros(shape, dtype=torch.float32) 
    elif method == "ones":
        x = torch.ones(shape, dtype=torch.float32)
    elif method == "full":
        x = torch.full(shape, value, dtype=torch.float32)
    else:
        raise Exception(f"invalid method {method}")
    return x.tolist()