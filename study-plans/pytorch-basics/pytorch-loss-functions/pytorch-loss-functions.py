import torch

def compute_mse_loss(pred, target):
    squared_diff = (pred - target) ** 2
    return torch.mean(squared_diff).item()

def compute_cross_entropy_loss(pred, target):
    # The naive approach of computing softmax then log can overflow when logits are large. 
    # The log-sum-exp trick subtracts the row-wise maximum before exponentiating, which prevents numerical overflow without changing the result. 
    # The loss for each sample is the log-sum-exp minus the logit corresponding to the correct class.
    target_t = torch.tensor(target, dtype=torch.long)
    max_value = pred.max(dim=1, keepdim=True).values 
    shifted = pred - max_value
    log_sum_exp = shifted.exp().sum(dim=1).log() + max_value.squeeze(1)
    correct_logits = pred[torch.arange(pred.shape[0]), target_t]
    return (log_sum_exp - correct_logits).mean().item()

def compute_huber_loss(pred, target, delta=1.0):
    target_t = torch.tensor(target, dtype=torch.float32)
    diff = (pred - target_t).abs()
    loss = torch.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
    return loss.mean().item()
    
def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred = torch.tensor(pred, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    if method == "mse":
        return compute_mse_loss(pred, target)
    elif method == "cross_entropy":
        return compute_cross_entropy_loss(pred, target)
    elif method == "huber":
        return compute_huber_loss(pred, target, delta)
    raise Exception(f"invalid loss method {method}")
    
