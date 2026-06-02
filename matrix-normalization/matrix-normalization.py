import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    matrix = np.array(matrix, dtype=float)
    if matrix.ndim != 2 or (axis != None and axis > 1):
        return None
    if norm_type == "l1":
        norms = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == "l2":
        norms = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
    elif norm_type == "max":
        norms = np.max(np.abs(matrix), axis=axis, keepdims=True)
    else:
        return None
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms