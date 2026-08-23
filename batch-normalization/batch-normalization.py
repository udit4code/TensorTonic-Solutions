import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """Return the training-time BatchNorm output."""
    # Step 0: Convert python data structures into numpy data structures 
    np_x = np.asarray(x, dtype=np.float64)
    np_gamma = np.asarray(gamma, dtype=np.float64)
    np_beta = np.asarray(beta, dtype=np.float64)
    normalized = None 
    # Step 1 : Decide the axes for reduction 
    if np_x.ndim == 2:
        # Case 1 : If shape of np_x is (N, D)
        # BatchNorm has 1 gamma and 1 beta for every feature. 
        # So, dimension of gamma and beta initially in this case is (D, )
        # For gamma and beta to be applied, it needs to be reshape to (1, D) for broadcasting.
        # So, (N,D) vs (1, D) = (N, D)
        N, D = np_x.shape
        axes = (0,)
        parameter_shape = (1, -1)
    else:
        # Case 2 : If shape of np_x is (N, C, H, W), which means (batch_size, channel_count, height of channel, width of channel)
        # In this case, BatchNorm has 1 gamma and 1 beta per channel. 
        # So, dimension of gamma and beta initially is (C,) . 
        # For broadcasting with (N, C, H, W), we want gamma and beta to be made (1, C, 1, 1). 
        # So, we set paramter shape to (1, -1, 1, 1) and the -1 means pick the last dimension (C in this case) from initial shape pf gamma and beta. 
        N, C, H, W = np_x.shape
        axes = (0, 2, 3,)
        parameter_shape = (1, -1, 1, 1)
    # Step 2 : Compute mean over chosen axis 
    mean = np.mean(np_x, axis=axes, keepdims=True)
    # Step 3 : Compute variance over chosen axis 
    variance = np.var(np_x, axis=axes, keepdims=True)
    # Step 4 : Apply normalization over mean and variance 
    normalized = (np_x - mean) / np.sqrt(variance + eps)
    # Step 5 : Reshape gamma and beta 
    gamma = np.reshape(gamma, parameter_shape)
    beta = np.reshape(beta, parameter_shape)
    # Step 6 : Apply normalization 
    normalized = normalized * gamma + beta 
    return normalized 