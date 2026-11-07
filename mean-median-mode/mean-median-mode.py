import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    if not x:
        return {"mean": None, "median": None, "mode": None}
    
    # Step 1 : Calculate mean and median using numpy
    mean_val = float(np.mean(x))
    median_val = float(np.median(x))
    
    # Step 2 : Calculate mode(s) 
    frequency_map = {}
    for x_item in x:
        if x_item not in frequency_map:
            frequency_map[x_item] = 0
        frequency_map[x_item] += 1
    max_freq = max(frequency_map.values())
    modes = sorted([k for k, v in frequency_map.items() if v == max_freq])
    mode_val = float(modes[0])

    return {
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val
    }