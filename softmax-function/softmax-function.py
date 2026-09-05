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
    # Step 2 : Compute max of x along the last dimension 
    max_x = np.max(x, axis=1, keepdims=True)
    # Step 3 : Shift x by max_x
    shifted_x = x - max_x
    # Step 4 : Exponentiate shifted_x 
    exp_values = np.exp(shifted_x)
    # Step 4 : Get total sum
    sum = np.sum(exp_values, axis=1, keepdims=True)

    result = exp_values / sum 
    return result 