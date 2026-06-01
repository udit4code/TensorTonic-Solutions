import numpy as np

def matrix_trace_v1(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A_np = np.array(A, dtype=np.float64)
    return np.trace(A_np)

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A_np = np.array(A, dtype=np.float64)
    return np.sum(np.diagonal(A_np))