import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Step 1 : Convert x to np.ndarray
    x = np.asarray(x, dtype=np.float64)
    # Step 1.1 : Handle special case of when x.ndim = 1
    if x.ndim == 1:
        shifted = x - np.max(x)
        exp_values = np.exp(shifted)
        sum = np.sum(exp_values)
        return exp_values / sum
    # Step 2 : Compute max of x  
    # axis=1 means apply Softmax along each row for a 2D array. 
    # if x.shape is (2, 3), then, it means it has 2 rows and 3 columns.
    # Now, np.max(x, axis=1) means that it takes the max across columns (across axis=1) for each row. 
    # keepdims=True preserves the reduced dimension, so that shape of max_x is (2, 1) instead of (2,).
    # This helps in broadcasting in later stages.
    max_x = np.max(x, axis=1, keepdims=True)
    # Step 3 : Shift x by max_x
    shifted_x = x - max_x
    # Step 4 : Exponentiate shifted_x 
    exp_values = np.exp(shifted_x)
    # Step 4 : Get total sum
    sum = np.sum(exp_values, axis=1, keepdims=True)

    result = exp_values / sum 
    return result 