import numpy as np

def pairwise_diff(a):
    """Returns: np.ndarray of shape (n, n) where out[i,j] = a[i] - a[j]"""
    # Assume row_vector_a has dimension (1, n)
    row_vector_a = np.array(a, dtype=np.float64)
    # Step 1 : Convert row_vector_a into a column_vector_a whose dimension is (n, 1)
    column_vector_a = np.reshape(row_vector_a, (len(a), 1))
    # Step 2 : Apply broadcasting between column_vector_a and row_vector_a
    # Why ? (1, n) - (n, 1) => By rules of broadcasting, (max(1, n), max(n, 1)) = (n, n)
    # Example : row_vector_a = [1, 2] (dim = 1 x 2) -> [[1, 2], [1, 2]] (dim = 2 x 2) after broadcasting
    # col_vector_a = [[1], [2]] (dim = 2 x 1) -> [[1, 1], [2, 2]] (dim = 2 x 2) after broadcasting
    # So, [[1, 2], [1, 2]] - [[1, 1], [2, 2]] = [[0, 1], [-1, 0]]
    pairwise_diff = column_vector_a - row_vector_a
    return pairwise_diff
    