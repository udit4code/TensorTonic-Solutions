from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    if not x:
        return {"mean": None, "median": None, "mode": None}
    
    # Calculate mean and median using numpy
    mean_val = float(np.mean(x))
    median_val = float(np.median(x))
    
    # Calculate mode(s) using Counter
    counts = Counter(x)
    max_freq = max(counts.values())
    modes = sorted([k for k, v in counts.items() if v == max_freq])
    # If there is a single mode, return it as a scalar; otherwise, return the list of modes
    mode_val = float(modes[0])

    return {
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val
    }