import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    # For very large negative values, np.exp(-x) can overflow.
    x = np.array(x, dtype=np.float64)
    # np.where(condition, value_if_true, value_if_false) is NumPy's vectorized version of:
    # if condition:
    #   value_if_true
    # else:
    #   value_if_false
    #
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))