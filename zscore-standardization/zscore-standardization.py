import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    # Write code here
    try:
        data = np.array(X, dtype=np.float64)
        if data.size == 0:
            return None
        if axis is not None and (axis < -data.ndim or axis >= data.ndim):
            return None
            
        mean = np.mean(data,axis=axis,keepdims=True)
        std = np.std(data,axis=axis,keepdims=True)
        return (data - mean) / (std + eps)

    except (TypeError, ValueError):
        return None
