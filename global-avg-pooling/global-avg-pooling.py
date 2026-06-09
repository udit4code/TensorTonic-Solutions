import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.asarray(x)
    if x.ndim == 3:
        # Average over H and W
        return np.mean(x, axis=(1, 2))
    elif x.ndim == 4:
        # Average over H and W
        return np.mean(x, axis=(2, 3))
    raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")