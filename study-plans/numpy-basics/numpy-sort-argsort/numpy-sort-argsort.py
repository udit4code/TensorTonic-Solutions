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
        # Why ? Because we want to do broadcasting.
        # Eg: np_data = [[1, 2], [3, 4]], its shape is (2, 2) and its size is 4.
        # We have np.arange(np_data.shape[0]) = np.arange(2) = [0, 1] -> shape (2, ) and size 2.
        # Now, reshape(-1, 1) means we want r rows and 1 column. So, r x 1 = size 2 => r = 2
        # So, we convert it into a 2 x 1 column vector
        # So, we have np.arange(2).reshape(-1, 1) which leads to [[0], [1]]
        # We use a column vector so NumPy can broadcast it against the indices matrix.
        rows = np.arange(np_data.shape[0]).reshape(-1, 1)
        # So, here, via broadcasting, rows go from [[0], [1]] to [[0, 0], [1, 1]] 
        # Now, we have np_data[ rows = [[0, 0], [1, 1]] ,  indices = [[1, 0], [0, 1]] ]
        # This translates to np_data[i][j] = np_data[rows[i][j]][indices[i][j]]
        # So, np_data[0][0] = np_data[rows[0][0]][indices[0][0]] = np_data[0][1] = 2
        # np_data[0][1] = np_data[rows[0][1]][indices[0][1]] = np_data[0][0] = 1
        # np_data[1][0] = np_data[rows[1][0]][indices[1][0]] = np_data[1][0] = 3
        # np_data[1][1] = np_data[rows[1][1]][indices[1][1]] = np_data[1][1] = 4 
        # So, sorted_data = [[2, 1], [3, 4]]
        sorted_data = np_data[rows, indices]
    else:
        raise ValueError("axis must be 0 or 1 for a 2D array")

    return np.stack([sorted_data, indices])