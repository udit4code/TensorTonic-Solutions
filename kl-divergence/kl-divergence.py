import numpy as np

# Think of kl_divergence as a fundamental statistic measure of how one 
# probability distribution differs from a second reference distribution
def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Convert input lists to NumPy arrays
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    
    # Step 1: Clip q to prevent division by zero or log(0)
    q_arr = np.clip(q_arr, eps, 1.0) 

    # Step 2: Handle entries where p is zero
    # Mathematically, as p -> 0, p * log(p / q) -> 0. 
    # We filter out p == 0 elements to avoid invalid log operations.
    mask = p_arr > 0
    p_filtered = p_arr[mask]
    q_filtered = q_arr[mask]

    # Step 3: Compute the sum of p * log(p / q)
    kl_value = np.sum(p_filtered * np.log(p_filtered / q_filtered))

# Step 4: Cast to a native Python float
    return float(kl_value.item())