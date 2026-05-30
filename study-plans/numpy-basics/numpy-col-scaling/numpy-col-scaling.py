import numpy as np

def scale_cols(data, weights):
    """Returns: np.ndarray of shape (m, n), each column scaled by corresponding weight"""
    np_data = np.array(data, dtype=np.float64)
    np_weights = np.array(weights, dtype=np.float64)
    # Step 1 : np_data has dim (m, n) and np_weights has dim (1, n) 
    # So, by rules of broadcasting, (Max(m, 1) , Max(n, n)) = (m, n)
    return np_data * np_weights
    