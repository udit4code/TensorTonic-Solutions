import numpy as np

def outer_sum_v1(a, b):
    """Returns: np.ndarray of shape (m, n), outer sum where out[i,j] = a[i] + b[j]"""
    np_a = np.array(a, dtype=np.float64)
    np_b = np.array(b, dtype=np.float64)
    # Step 1 : np_a has a shape (m, ) and np_b has a shape (n, ). So, we need to reshape at least one of them
    np_a_reshaped = np.reshape(np_a, (-1, 1))
    # Step 2 : np_a_reshaped is in (m, 1), so, now we can apply broadcasting.
    return np_a_reshaped + np_b

def outer_sum_v2(a, b):
    """Returns: np.ndarray of shape (m, n), outer sum where out[i,j] = a[i] + b[j]"""
    np_a = np.array(a, dtype=np.float64)
    np_b = np.array(b, dtype=np.float64)
    return np.add.outer(np_a, np_b)



def outer_sum(a, b):
    """Returns: np.ndarray of shape (m, n), outer sum where out[i,j] = a[i] + b[j]"""
    return outer_sum_v2(a, b)