import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    assert len(y_true) == len(y_pred), f"y_pred and y_true have different lengths {len(y_true)} and {len(y_pred)}"
    n = len(y_true)
    # Step 1 : Convert y_true and y_pred to numpy arrays 
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    # Step 2 : Compute numerator 
    residual = (y_true - y_pred) * (y_true - y_pred)
    residual_sum = np.sum(residual)

    # Step 3 : Compute denominator
    mean_target = np.sum(y_true) / n 
    squares = (y_true - mean_target) * (y_true - mean_target)
    squares_sum = np.sum(squares)

    if squares_sum == 0:
        if residual_sum == 0:
            return 1.0 
        else:
            return 0.0

    output = 1 - (residual_sum / squares_sum)

    return float(output)