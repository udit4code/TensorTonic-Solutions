import numpy as np

def summarize(data, axis):
    """Returns: np.ndarray of shape (4, k), rows are mean, std, min, max"""  
    np_data = np.array(data, dtype=np.float64)
    mean = np.mean(np_data, axis=axis)
    std = np.std(np_data, axis=axis)
    min = np.min(np_data, axis=axis)
    max = np.max(np_data, axis=axis)
    return np.stack([mean, std, min, max])