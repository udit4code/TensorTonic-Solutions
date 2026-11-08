import numpy as np

def get_gini(y: np.ndarray) -> float:
    """Helper function to calculate Gini impurity for a single array."""
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / n
    return 1.0 - np.sum(probabilities ** 2)

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the weighted Gini impurity of a split as a float.
    """
    y_left = np.array(y_left)
    y_right = np.array(y_right)
    
    n_left = len(y_left)
    n_right = len(y_right)
    total_n = n_left + n_right
    
    if total_n == 0:
        return 0.0
        
    # Step 1 : Calculate Gini impurity for each child node
    gini_l = get_gini(y_left)
    gini_r = get_gini(y_right)
    
    # Step 2 : Compute the weighted average impurity of the split
    weighted_gini = (n_left / total_n) * gini_l + (n_right / total_n) * gini_r
    
    return float(weighted_gini)