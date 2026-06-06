import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    result = None 
    x = torch.tensor(x, dtype=torch.float32)
    if op == "flatten":
        result = torch.flatten(x)  
    elif op == "squeeze":
        result  = torch.squeeze(x)
    elif op == "transpose":
        result = x.T
    else:
        raise Exception(f"invalid op {op}")
    return result 
