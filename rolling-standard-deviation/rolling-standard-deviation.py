import math 
import numpy as np

def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    # Write code here
    w_start = 0
    n = len(values)
    rolling_std_values = [ ]
    while w_start < n - window_size + 1:
        # Extract the window first
        w_end = w_start + window_size - 1
        window = values[w_start : (w_end + 1)]
        np_window = np.array(window)
        # Compute the rolling statistics
        window_mean = np.mean(np_window)
        window_std = np.sqrt(np.sum((np_window - window_mean) ** 2) / window_size)
        rolling_std_values.append(window_std)
        # For Next iteration
        w_start += 1
    return rolling_std_values