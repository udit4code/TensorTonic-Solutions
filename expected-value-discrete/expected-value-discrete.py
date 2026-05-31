import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x, dtype=np.float64)
    p = np.array(p, dtype=np.float64)
    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")
    if abs(np.sum(p) - 1.0) > 1e-6:
        raise ValueError("probabilities must sum to 1")
    return np.dot(x, p) # Alternate : np.sum(x * p)
