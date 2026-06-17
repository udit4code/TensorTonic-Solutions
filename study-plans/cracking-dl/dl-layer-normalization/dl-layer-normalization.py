import numpy as np

def layer_normalization(x, gamma, beta, eps=1e-5, mode="forward", d_output=None):
    """
    Returns: Dict with "output", "mean", "var", "x_hat", and optionally "dx", "dgamma", "dbeta".
    """
    decimal_places = 4
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2, "x does not have 2 dimensions"
    N, D = x.shape
    # Step 1 : Get mean for a given sample across each column/features
    mean = (np.sum(x, axis=1)) / D
    # Step 2 : Get variance for a given sample across each column/features
    var = (np.sum((x - mean.reshape(-1, 1)) ** 2, axis=1)) / D
    # Step 3 : Get x_hat from mean and var
    x_hat = (x - mean.reshape(-1, 1)) / np.sqrt(var.reshape(-1, 1) + eps)
    # Step 4 : Get output from x_hat, gamma and beta.
    output = gamma * x_hat + beta
    result = {
        "output" : np.round(output, decimal_places),
        "mean" : np.round(mean, decimal_places),
        "var" : np.round(var, decimal_places),
        "x_hat" : np.round(x_hat, decimal_places),
    }
    # Step 5 : [Optiona] Compute gradients
    if mode == "backward" and d_output:
        d_output = np.array(d_output, dtype=np.float64)
        dgamma = np.sum(d_output * x_hat, axis=0)
        dbeta = np.sum(d_output, axis=0)
        dx_hat = d_output * gamma
        dx_hat_mean = np.mean(dx_hat, axis=-1, keepdims=True)
        dx_hat_xhat_mean = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        dx = (1.0 / std.reshape(-1, 1)) * (dx_hat - dx_hat_mean - x_hat * dx_hat_xhat_mean)
        result["dx"] = np.round(dx, decimal_places)
        result["dgamma"] = np.round(dgamma, decimal_places)
        result["dbeta"] = np.round(dbeta, decimal_places)
    return result