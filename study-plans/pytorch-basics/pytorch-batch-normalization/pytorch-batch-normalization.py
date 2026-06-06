import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    X = torch.tensor(X, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=X.dtype)
    beta = torch.tensor(beta, dtype=X.dtype)

    # Step 1 : Get Column-wise mean
    mean = torch.mean(X, dim=0)
    # Step 2 : Column-wise population variance
    # torch.var() defaults to unbiased=True, which computes sample variance (N-1 denominator). BatchNorm uses population variance (N denominator), i.e. unbiased=False.
    var = torch.var(X, dim=0, unbiased=False)
    # Step 3 : Normalize with mean and variance
    x_hat = (X - mean) / torch.sqrt(var + eps)
    # Scale and shift
    y = gamma * x_hat + beta
    return y
