import numpy as np

def get_entropy(y: np.ndarray) -> float:
    """Helper function to calculate Shannon entropy."""
    if len(y) == 0:
        return 0.0
    # np.unique(...) finds the unique elements in an array and count how many times each unique element appears.
    unique_values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # Filter out zero probabilities to avoid log2(0)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    y = np.asarray(y)
    split_mask = np.array(split_mask, dtype=bool)
    
    n = len(y)
    if n == 0:
        return 0.0
        
    # Step 1 : Get Parent Node's entropy
    parent_entropy = get_entropy(y)
    
    # Step 2 : Split the dataset via split_mask
    left_y = y[split_mask]
    right_y = y[~split_mask]
    
    n_left = len(left_y)
    n_right = len(right_y)
    
    # If a split results in an empty subset, information gain is 0
    if n_left == 0 or n_right == 0:
        return 0.0
        
    # Step 3 : Get weighted average entropy of the children nodes
    child_entropy = (n_left / n) * get_entropy(left_y) + (n_right / n) * get_entropy(right_y)
    
    # Information gain is the reduction in entropy
    return float(parent_entropy - child_entropy)