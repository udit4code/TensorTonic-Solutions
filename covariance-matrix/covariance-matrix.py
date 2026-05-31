import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.array(X, dtype=np.float64)
    if X.ndim != 2:
        return None
    N, D = X.shape
    if N < 2:
        return None
    # Step 1: Compute column means.
    # Example: [[1,2], [2,4], [3,6]] -> [(1 + 2 + 3)/3,(2 + 4 + 6)/3] = [2, 4]
    means = np.mean(X, axis=0)
    # Step 2: Center each column.
    # Example : [[1,2], [2,4], [3,6]] - [2,4] = Via broadcasting, [[2, 4], [2, 4], [2, 4]]
    # = [[1,2], [2,4], [3,6]] - [[2,4], [2,4], [2 4]]
    # = [[-1,-2],[0,0],[1,2]]
    centered = X - means
    # Step 3: Covariance matrix.
    # Cov = X_centered^T X_centered /(m-1), whose shape is (n,n)
    covariance = centered.T @ centered / (N - 1)
    return covariance