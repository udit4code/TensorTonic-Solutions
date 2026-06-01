import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    n = len(v)
    A = np.zeros((n, n), dtype=np.float64)
    indices = np.arange(n)
    A[indices, indices] = v 
    return A
