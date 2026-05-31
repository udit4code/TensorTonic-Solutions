import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""
    np_data = np.array(data, dtype=np.float64)
    # Step 1 : Get column_wise mean 
    column_mean = np.mean(np_data, axis=0)
    # Step 2 : Get column_wise std 
    column_std = np.std(np_data, axis=0)
    # Via broadcasting, (n, ) will become (m, n) so that it gets broadcasted to all rows
    return (np_data - column_mean) / column_std