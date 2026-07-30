import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # Step 1 : Convert x, W1, W2 into numpy ndarray
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    # Step 2 : Take a copy of x. We want to create a completely independent duplicate, with its own allocated memory 
    x_copy = x.copy()
    # Step 3 : Apply ReLU(x @ W1.T) to get h
    h = np.maximum(0, x @ W1.T) 
    # Step 4 : Apply ReLU(h @ W2.T + x_copy)
    y = np.maximum(0, h @ W2.T + x_copy)

    return y.tolist()
