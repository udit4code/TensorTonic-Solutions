import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    y = np.asarray(y)
    if num_classes is None:
        num_classes = np.max(y) + 1
    # Create all zeros
    result = np.zeros((len(y), num_classes), dtype=np.float32)

    # Advanced indexing:
    # rows:    [0,1,2,...N-1]
    # columns: labels
    # set those positions to 1
    result[np.arange(len(y)), y] = 1.0
    return result
    