import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    a_dot_b = np.sum(a * b) 
    if np.isclose(a_dot_b, 0.0):
        return 0.0
    l2_norm = lambda x : np.sqrt(np.sum(x * x))
    norm_b = l2_norm(b)
    norm_a = l2_norm(a)
    if np.isclose(norm_a, 0.0) or np.isclose(norm_b, 0.0):
        return 0.0
    return a_dot_b / (norm_a * norm_b)
    