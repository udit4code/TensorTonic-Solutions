import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    result = None
    if op == "add":
        result = x + y 
    elif op == "multiply":
        result = x * y 
    elif op == "matmul":
        result = x @ y 
    elif op == "power":
        result = x ** y
    elif op == "max":
        result = torch.maximum(x, y)
    return result.tolist()
        