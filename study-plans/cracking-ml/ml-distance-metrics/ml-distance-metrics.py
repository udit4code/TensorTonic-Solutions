import math
import numpy as np 

def get_euclidean_distance(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    squared_diff = (x - y) ** 2
    return np.sqrt(np.sum(squared_diff))

def get_norm(x):
    return np.sqrt(np.sum(x ** 2))
    
def get_cosine_distance(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    x_norm = get_norm(x)
    y_norm = get_norm(y)
    if np.isclose(x_norm, 0.0, atol=1e-9) or np.isclose(x_norm, 0.0, atol=1e-9):
        return 0.0
    dot_product = np.sum(x * y)
    return 1 - (dot_product/(x_norm * y_norm))

def get_minkowski_distance(x, y, p=2):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    # Compute |x_i - y_i|^p for each dimension
    diff = np.abs(x - y) ** p
    # Sum across all dimensions
    total = np.sum(diff)
    # Take the p-th root
    return total ** (1 / p)
    
def get_chebyshev_distance(x, y):
    # Convert inputs to NumPy arrays
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    # Compute absolute difference in each dimension
    diff = np.abs(x - y)
    # Return the largest difference
    return int(np.max(diff))

def get_manhattan_distance(x, y):
    d = get_minkowski_distance(x, y, p=1)
    return int(d)
    
    
def distance_metric(x, y, metric, p=2):
    """
    Compute the distance between vectors x and y using the specified metric.
    Returns: float rounded to 4 decimal places
    """
    result = None
    if metric == "euclidean":
        result = get_euclidean_distance(x, y) 
    elif metric == "cosine":
        result = get_cosine_distance(x, y)  
    elif metric == "minkowski":
        result = get_minkowski_distance(x, y, p) 
    elif metric == "chebyshev":
        result = get_chebyshev_distance(x, y)  
    elif metric == "manhattan":
        result = get_manhattan_distance(x, y)  
    else:
        raise Exception(f"invalid metric : {metric}")
    return result