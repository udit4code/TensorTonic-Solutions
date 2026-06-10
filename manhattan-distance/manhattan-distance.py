import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return np.sum(np.abs(x - y))