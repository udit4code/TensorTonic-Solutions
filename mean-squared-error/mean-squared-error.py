import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_pred.shape == y_true.shape, f"shapes of y_pred {y_pred.shape} and y_true {y_true.shape} are not equal"
    N = y_pred.shape[0]
    return (np.sum((y_pred - y_true) ** 2)) / N 