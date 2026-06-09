import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return np.sqrt(np.sum((x - y) ** 2))