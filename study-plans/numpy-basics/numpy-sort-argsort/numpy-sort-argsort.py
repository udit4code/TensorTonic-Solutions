import numpy as np

def sort_with_indices_v1(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    np_data = np.array(data, dtype=np.float64)
    indices = np.argsort(np_data, axis=axis)
    # We use the cleanest way : use numpy's take_along_axis()
    sorted_data = np.take_along_axis(np_data, indices, axis=axis)
    return np.stack([sorted_data, indices])

def sort_with_indices_v2(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    np_data = np.array(data, dtype=np.float64)
    # np.argsort() the indices that would sort an array
    indices = np.argsort(np_data, axis=axis)
    # Here, we sort it twice, but use 2 passes of sorting : once via argsort() and the other via sort()
    sorted_data = np.sort(np_data, axis=axis)
    return np.stack([sorted_data, indices])

def sort_with_indices(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    np_data = np.asarray(data, dtype=np.float64)
    indices = np.argsort(np_data, axis=axis)

    if axis == 0:
        cols = np.arange(np_data.shape[1])
        sorted_data = np_data[indices, cols]
    elif axis == 1:
        # rows = np.arange(np_data.shape[0])[:, None]
        rows = np.arange(np_data.shape[0]).reshape(-1, 1)
        sorted_data = np_data[rows, indices]
    else:
        raise ValueError("axis must be 0 or 1 for a 2D array")

    return np.stack([sorted_data, indices])