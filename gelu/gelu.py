import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Step 1 : Convert x to numpy array 
    x_arr = np.asarray(x, dtype=np.float64)
    # Step 2: Vectorize math.erf for standard numpy arrays
    # (Or use scipy.special.erf(x_arr / np.sqrt(2.0)) if scipy is available)
    erf_vec = np.vectorize(math.erf)
    # Step 3 : Compute GELU using the exact CDF formula
    cdf = 0.5 * (1.0 + erf_vec(x_arr / math.sqrt(2.0)))
    
    y = x_arr * cdf 
    return y.astype(float)
