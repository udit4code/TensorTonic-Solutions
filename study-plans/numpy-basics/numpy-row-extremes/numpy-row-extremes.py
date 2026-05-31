import numpy as np

def row_extremes(data):
    """Returns: np.ndarray of shape (4, m), rows are max_val, max_col, min_val, min_col"""
    np_data = np.asarray(data, dtype=np.float64)

    # Step 1 : Get the max of each row
    max_val = np.max(np_data, axis=1)
    # Step 2 : Get the index of maximum column of each row
    max_col = np.argmax(np_data, axis=1)

    # Step 3 : Get the min of each row
    min_val = np.min(np_data, axis=1)
    # Step 4  :Get the index of minimum column of each row
    min_col = np.argmin(np_data, axis=1)

    return np.stack([max_val, max_col, min_val, min_col])