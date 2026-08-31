import numpy as np 

def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    """
    Returns sampling probabilities and normalized importance weights.
    """
    N = len(priorities)
    np_priorities = np.asarray(priorities, dtype=np.float64)
    np_alpha = np.float64(alpha)
    np_beta = np.float64(beta)
    
    # Step 1 : Compute Powered Priorities 
    powered_priorities = np_priorities ** np_alpha 
    sum_priorities = np.sum(powered_priorities)
    
    # Step 2 : Compute Sampling Probabilities
    probabilities = powered_priorities / sum_priorities 

    # Step 3 : Compute raw importance sampling weights 
    weights = (N * probabilities)** (-np_beta)
    max_weight = np.max(weights)

    # Step 4 : Normalize weights by max weight 
    normalized_weights = weights / max_weight

    result = [probabilities.tolist(), normalized_weights.tolist()]
    return result