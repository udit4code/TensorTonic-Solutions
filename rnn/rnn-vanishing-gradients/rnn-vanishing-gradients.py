import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
        Simulate gradient norm decay/explosion over T backward steps.
    
        Returns:
            list of length T
    """

    spectral_norm = np.linalg.norm(W_hh, ord=2)
    norms = [1.0]
    grad_norm = 1.0
    for _ in range(1, T):
        grad_norm *= spectral_norm
        norms.append(float(grad_norm))
    return norms