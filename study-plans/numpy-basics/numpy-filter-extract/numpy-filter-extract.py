import numpy as np

def filter_and_extract(data, row_start, row_stop, threshold):
    """
    Returns: 1D ndarray of float64
    """
    np_data = np.array(data, dtype=np.float64)
    filtered_np_data = np_data[row_start: row_stop, :]
    return filtered_np_data[filtered_np_data > threshold]