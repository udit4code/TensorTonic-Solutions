import math
import numpy as np 

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    # Step 1 : Compute bound 
    bound = math.sqrt(6 / fan_in)
    # Step 2 : Map each raw weight from [0, 1] to [-L, L]
    W_np = np.asarray(W, dtype=np.float64)
    W_np = W_np * 2 * bound - bound

    # Step 3 : Convert the numpy array back to a standard Python list object
    return W_np.tolist()