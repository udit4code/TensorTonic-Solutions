import numpy as np 

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    x = np.array(values, dtype=np.float64)
    min_x = np.min(x)
    max_x = np.max(x)
    w = ( max_x - min_x ) / num_bins
     # All values are identical
    if np.isclose(w, 0.0):
        return [0] * len(x)
    assigned_bins = [ ]
    for value in x:
        bin_of_value = min(np.floor((value - min_x) / w), num_bins - 1)
        assigned_bins.append(bin_of_value)
    return assigned_bins