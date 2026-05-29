import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    # Compute min along axis
    X_min = np.min(X,axis=axis,keepdims=True)
    # Compute max along axis
    X_max = np.max(X,axis=axis,keepdims=True)
    denominator = X_max - X_min
    # avoid divide by zero
    scaled = ((X - X_min)/ (denominator + eps))
    return scaled