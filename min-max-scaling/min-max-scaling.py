import numpy as np 
def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    eps = 1e-12
    X = np.asarray(data, dtype=float)
    rows = len(X)
    cols = len(X[0])
    result = np.zeros_like(X)
    for col in range(cols):
        column = X[:, col]
        col_min = np.min(column)
        col_max = np.max(column)
        denominator = col_max - col_min
        result[:, col] = ((column - col_min)/(denominator + eps))
    return result.tolist()