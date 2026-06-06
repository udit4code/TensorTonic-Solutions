import torch

def compute_mse_loss(pred, target):
    # Compute element-wise prediction error.
    #
    # Example:
    # pred   = [2, 4, 6]
    # target = [1, 5, 7]
    #
    # pred - target
    # = [1, -1, -1]
    squared_diff = (pred - target) ** 2

    # Square the errors so:
    #  - negative and positive errors contribute equally
    #  - larger mistakes are penalized more heavily
    #
    # [1, -1, -1]
    # ->
    # [1, 1, 1]

    # Average across all elements.
    #
    # mean([1,1,1])
    # = 1.0
    return torch.mean(squared_diff).item()

def compute_cross_entropy_loss(pred, target):

    # Convert class labels into integer indices.
    #
    # Example:
    # target = [0, 2, 1]
    #
    # means:
    # sample0 -> class0
    # sample1 -> class2
    # sample2 -> class1
    target_t = torch.tensor(target, dtype=torch.long)
    # Find largest logit in each row.
    # Example: pred = [[2.0, 1.0, 0.1]]
    # max_value = [[2.0]]
    # keepdim=True keeps shape (batch_size,1)
    max_value = pred.max(dim=1, keepdim=True).values
    # Shift logits before exponentiation.
    # Original: [2.0,1.0,0.1]
    # Shifted: [2.0,1.0,0.1] - [2.0, 2.0, 2.0] = [0,-1,-1.9]
    # This prevents overflow in exp().
    shifted = pred - max_value
    # Compute: log(sum(exp(logits))) using the numerically stable log-sum-exp trick.
    # Example:
    # exp([0,-1,-1.9]) = [1,0.3679,0.1496]
    # sum = 1.5175
    # log(sum)=0.417
    # add max_value back: 0.417 + 2.0 = 2.417
    log_sum_exp = shifted.exp().sum(dim=1).log() + max_value.squeeze(1)
    # Extract the logit corresponding to the correct class.
    # Example:
    # target = [0]
    # pred[0,0] = 2.0
    correct_logits = pred[torch.arange(pred.shape[0]),target_t]
    # Cross Entropy: log(sum(exp(logits))) - correct_class_logit
    # Example: 2.417 - 2.0 = 0.417
    # Average across batch.
    return (log_sum_exp - correct_logits).mean().item()
    


def compute_huber_loss(
    pred,
    target,
    delta=1.0
):
    # Convert target into float tensor.
    target_t = torch.tensor(
        target,
        dtype=torch.float32
    )

    # Compute absolute error.
    # Example:
    # pred   = [1,4,10]
    # target = [1,5,5]
    # diff = [0,1,5]
    diff = (pred - target_t).abs()

    # Piecewise loss.
    #
    # If error is small:
    # 0.5 * diff^2 behaves like MSE.
    #
    # If error is large:
    # delta * (diff - 0.5*delta) behaves like MAE.
    #
    # This prevents huge outliers from dominating training.
    loss = torch.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
    # Average loss across all samples.
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
    
