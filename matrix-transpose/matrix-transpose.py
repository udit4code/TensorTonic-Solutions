import numpy as np

def matrix_transpose_v1(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A_np = np.array(A, dtype=np.float64)
    return A_np.T

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A_np = np.array(A, dtype=np.float64)
    m, n = A_np.shape
    transpose_A = np.zeros((n, m), dtype=np.float64)
    for row_idx in range(m):
        for col_idx in range(n):
            transpose_A[col_idx][row_idx] = A_np[row_idx][col_idx]
    return transpose_A