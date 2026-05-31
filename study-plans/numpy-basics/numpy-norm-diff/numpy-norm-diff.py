import numpy as np

def get_scaled_and_clipped_data(data, lo, hi):
    # Step 1 : Clip the data matrix in the range [lo, hi]
    clipped_data = np.clip(data, lo, hi)
    # Step 2 : Then, scale the data matrix , such that (x - lo)/(hi - lo) where x = clipped_data[i][j] . We will do broadcasting of scalers.
    # lo or hi is a scalar, whose shape is (). So, numpy conceptually prepends dimensions of size 1 as (1, 1). Now, (m, n) - (1, 1) = Via Broadcasting, (m, n) - (m, n)
    scaled_data = (clipped_data - lo)/(hi - lo)
    return scaled_data
    
def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""
    a_data = np.array(a, dtype=np.float64)
    b_data = np.array(b, dtype=np.float64)
    a_scaled_and_clipped = get_scaled_and_clipped_data(a_data, lo, hi)
    b_scaled_and_clipped = get_scaled_and_clipped_data(b_data, lo, hi)
    return np.abs(a_scaled_and_clipped - b_scaled_and_clipped)
    