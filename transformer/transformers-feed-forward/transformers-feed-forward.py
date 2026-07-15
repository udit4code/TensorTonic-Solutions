import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    hidden = np.maximum(0, x @ W1 + b1)  # ReLU
    output = hidden @ W2 + b2
    return output