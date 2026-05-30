import numpy as np

def extract_subarray(arr, row_start, row_stop, col_start, col_stop):
    """
    Returns: 2D ndarray of float64
    """
    np_arr = np.array(arr, dtype=np.float64)
    return np_arr[row_start : row_stop, col_start : col_stop]
