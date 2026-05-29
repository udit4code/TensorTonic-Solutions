import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    # initialize parameters
    weights = np.zeros(n_features)
    bias = 0.0
    for epoch_id in range(epochs):
        # Forward pass
        y_pred = (X @ weights + bias)
        error = y_pred - y
        # Gradients
        dw = (2 / n_samples * (X.T @ error))
        db = (2 / n_samples * np.sum(error))
        # Update parameters
        weights -= lr * dw
        bias -= lr * db

    return weights, bias
