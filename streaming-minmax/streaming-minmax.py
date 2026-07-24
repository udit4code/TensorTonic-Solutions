import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    return {
        "min": np.full(D, np.inf, dtype=np.float64),
        "max": np.full(D, -np.inf, dtype=np.float64),
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    # Step 0 : Safely convert input to a NumPy array if a list is passed
    X_batch = np.asarray(X_batch, dtype=np.float64)
    
    if X_batch.size == 0:
        return X_batch

    # Step 1 : Calculate batch boundaries along the instance tracking dimension
    batch_min = np.min(X_batch, axis=0)
    batch_max = np.max(X_batch, axis=0)

    # Step 2 : Perform running state adjustments
    state["min"] = np.minimum(state["min"], batch_min)
    state["max"] = np.maximum(state["max"], batch_max)

    # Step 3 : Scale the current incoming block using updated running limits
    range_delta = state["max"] - state["min"]
    normalized_batch = (X_batch - state["min"]) / (range_delta + eps)

    return normalized_batch