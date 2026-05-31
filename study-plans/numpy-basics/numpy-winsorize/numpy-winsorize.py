import numpy as np

def winsorize(data, lo_q, hi_q):
    """Returns: np.ndarray of shape (3, m, n), stacked clipped values, lo_mask, hi_mask"""
    # Assume shape of np_data is (m, n)
    np_data = np.array(data, dtype=np.float64)
    # Assume shape of low and high is (n, )
    low = np.percentile(np_data, lo_q, axis=0)
    high = np.percentile(np_data, hi_q, axis=0)
    clipped = np.clip(np_data, low, high)

    low_mask = np_data < low 
    high_mask = np_data > high 

    result = np.stack([clipped, low_mask, high_mask])
    return result