import numpy as np

def relu(x) -> np.ndarray:
    """Return ReLU applied elementwise to x."""
    x = np.asarray(x, dtype=np.float64)
    output = np.maximum(x, 0.0)
    return np.asarray(output, dtype=float) 