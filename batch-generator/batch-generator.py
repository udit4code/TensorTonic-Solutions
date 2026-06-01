import numpy as np

def get_batch_util(X, y, batch_size, drop_last):
    n_samples = len(X)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > n_samples and drop_last:
            break
        yield (X[start_idx:end_idx], y[start_idx:end_idx])


def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    n_samples = len(X)
    indices = np.arange(n_samples)
    if rng:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    return get_batch_util(X_shuffled, y_shuffled, batch_size,drop_last)