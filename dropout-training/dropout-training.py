import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Convert input list to a NumPy array
    arr = np.asarray(x, dtype=float)
    
    # Use default RNG if none is provided
    if rng is None:
        rng = np.random.default_rng()
        
    # Edge case: if drop probability is 1.0, everything is zeroed out
    if p >= 1.0:
        dropout_pattern = np.zeros_like(arr)
        output = np.zeros_like(arr)
        return output, dropout_pattern
        
    # Generate a binary mask where (1 - p) is the probability of keeping a unit
    # Inverted dropout scales active units by 1 / (1 - p)
    scale = 1.0 / (1.0 - p)
    dropout_pattern = (rng.random(arr.shape) >= p).astype(float) * scale
    
    # Apply mask to input array
    output = arr * dropout_pattern
    
    return output, dropout_pattern