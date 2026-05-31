import numpy as np

def sort_with_indices(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    np_data = np.array(data, dtype=np.float64)
    indices = np.argsort(np_data, axis=axis)
    sorted_data = np.take_along_axis(np_data, indices, axis=axis)
    return np.stack([sorted_data, indices])