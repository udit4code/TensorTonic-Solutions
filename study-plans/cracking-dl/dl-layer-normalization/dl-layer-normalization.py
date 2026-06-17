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
    output = gamma * x_hat + beta # y
    result = {
        "output" : np.round(output, decimal_places),
        "mean" : np.round(mean, decimal_places),
        "var" : np.round(var, decimal_places),
        "x_hat" : np.round(x_hat, decimal_places),
    }
    # Step 5 : [Optional] Compute gradients
    if mode == "backward" and d_output:
        # Think of d_output as dL/dy (Upstream gradient from the next layer)
        d_output = np.array(d_output, dtype=np.float64)
        # d_output[n, d] = ∂L/∂y[n, d]
        # where y = gamma * x_hat + beta
        d_output = np.array(d_output, dtype=np.float64)
        # Step 5.1 : Compute Gradient w.r.t gamma
        # Forward: y_i = gamma_i * x_hat_i + beta_i
        # Therefore: ∂y_i/∂gamma_i = x_hat_i
        # By chain rule: ∂L/∂gamma_i = Σ_batch (∂L/∂y_i)(∂y_i/∂gamma_i) = Σ_batch d_output_i * x_hat_i
        # The multiplication below is the Hadamard (element-wise) product;
        # and, the summation is performed over the batch dimension.
        dgamma = np.sum(d_output * x_hat, axis=0)
        # Step 5.2 : Gradient w.r.t beta
        # Since: y_i = gamma_i * x_hat_i + beta_i
        # we have: ∂y_i/∂beta_i = 1
        # Therefore: ∂L/∂beta_i = Σ_batch ∂L/∂y_i
        dbeta = np.sum(d_output, axis=0)
        # Step 5.3 : Gradient w.r.t x_hat
        # Since: y_i = gamma_i * x_hat_i + beta_i
        # we have: ∂y_i/∂x_hat_i = gamma_i
        # Therefore: ∂L/∂x_hat_i = (∂L/∂y_i)(∂y_i/∂x_hat_i) = d_output_i * gamma_i
        # Where, we are denoting dx_hat = ∂L/∂x_hat
        dx_hat = d_output * gamma
        # Step 5.4 : Final LayerNorm input gradient
        # We know: x_hat = (x - mean) / std, where both mean and variance depend on every feature of the sample.
        # Consequently, each input coordinate affects every normalized coordinate.
        # The fully-expanded Jacobian simplifies to:
        # dx = (1/std) * ( dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
        # 
        # We compute the two mean terms separately below.
        # mean(dx_hat) whose shape is (N, 1) represents:
        # (1/D) Σ_j dx_hat_j for each sample independently.
        dx_hat_mean = np.mean(dx_hat,axis=-1,keepdims=True)
        # mean(dx_hat * x_hat) whose shape is (N, 1) represents (1/D) Σ_j (dx_hat_j * x_hat_j) for each sample independently.
        dx_hat_xhat_mean = np.mean(dx_hat * x_hat,axis=-1,keepdims=True)
        # Standard deviation used in the forward pass:
        std = np.sqrt(var + eps)
        # Compact LayerNorm backward formula:
        # dx = (1/std) * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
        # Interpretation:
        # 1. dx_hat : Direct gradient contribution.
        # 2. -mean(dx_hat) : Removes the component that would change the mean.
        # 3. -x_hat * mean(dx_hat * x_hat) :Removes the component that would change the variance.
        #
        # Together these corrections ensure that the gradient respects the LayerNorm constraints:
        # mean(x_hat) = 0
        # var(x_hat)  = 1
        dx = (1.0 / std.reshape(-1, 1)) * (dx_hat - dx_hat_mean - x_hat * dx_hat_xhat_mean)
        result["dx"] = np.round(dx, decimal_places)
        result["dgamma"] = np.round(dgamma, decimal_places)
        result["dbeta"] = np.round(dbeta, decimal_places)
    return result