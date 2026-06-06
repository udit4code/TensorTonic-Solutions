import numpy as np

def get_minmax_scaling(X):
    min_x = np.min(X, axis=0)
    max_x = np.max(X, axis=0)

    numerator = X - min_x
    denominator = max_x - min_x
    result = np.zeros_like(X, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)

    return result
    

def get_standard_scaling(X):
    mean_x = np.mean(X, axis=0)
    std_x = np.std(X, axis=0)

    numerator = X - mean_x
    denominator = std_x
    result = np.zeros_like(X, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)

    return result
    
def feature_scale(X, method="minmax"):
    """
    Returns: 2D list of scaled values
    """
    X = np.array(X, dtype=np.float64)
    if method == "minmax":
        result = get_minmax_scaling(X)
    elif method == "standard":
        result = get_standard_scaling(X) 
    else:
        raise Exception(f"invalid input method {method}")
    return result
    