import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    data = np.array(arr, dtype=np.float64)
    if axis == 0:
        return data[indices, :]
    elif axis == 1:
        return data[:, indices]
    raise Exception(f"Invalid axis : {axis}")
    