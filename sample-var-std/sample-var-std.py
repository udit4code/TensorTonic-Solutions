import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    n = len(x)
    x = np.asarray(x)
    # Step 1 : Get mean 
    x_mean = np.mean(x)
    # Step 2 : Get diff with mean 
    diff = x - x_mean
    # Step 3 : Take dot product of diff with diff.T to get total
    total = np.dot(diff, diff)
    # Step 4 : Get unbiased sample variance 
    unbiased_sample_variance = total / (n - 1)
    std_dev = np.sqrt(unbiased_sample_variance)
    return {
        "variance" :  float(unbiased_sample_variance.item()),
        "standard_deviation" : float(std_dev.item())
    }