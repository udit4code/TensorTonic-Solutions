import numpy as np

def norm_gate(X, W, threshold):
    """Returns: np.ndarray of shape (n, k), gated projection where rows below threshold are zeroed"""
    np_X = np.array(X, dtype=np.float64)
    np_W = np.array(W, dtype=np.float64) 
    # Step 1 : Get Z = X @ W
    Z = np_X @ np_W
    # Step 2 : Compute L2 row of Each row.
    Z_copy = Z.copy()
    # Square Every Element 
    Z_copy = Z_copy ** 2
    # Then, sum across rows 
    L2_norms = np.sqrt(np.sum(Z_copy, axis=1))
    # Step 3 : Get the norm mask
    mask = L2_norms >= threshold
    mask = np.reshape(mask, (-1, 1))
    # Step 4 : Get the result by element-wise multiplying Z with mask
    result = Z * mask
    return result