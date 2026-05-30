import numpy as np

def scale_rows(data, weights):
    """Returns: np.ndarray of shape (m, n), each row scaled by corresponding weight"""
    np_data = np.array(data, dtype=np.float64)
    np_weights = np.array(weights, dtype=np.float64)
    # Key idea : Adding a new trailing dimension transforms the weight vector from shape (m, ) to (m, 1)
    np_weights = np.reshape(np_weights, np_weights.shape + (1,))
    return np_data * np_weights