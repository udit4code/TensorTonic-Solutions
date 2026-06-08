def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    N, d = X.shape 
    # Step 1 : Initialise w and b to 0 vector and 0 respectively.
    w = np.zeros(d)
    b = 0.0 

    # Step 2 : Run the training loop
    for epoch_id in range(epochs):
        # Step 2.1 : Forward pass 
        y_hat = X @ w + b 
        error = y_hat - y
        # Step 2.2 : Backward pass 
        dL_by_dw = (2.0 / N) * (X.T @ error) + 2 * alpha * w 
        dL_by_db = (2.0 / N) * np.sum(error)
        # Step 2.4 : Update w and b
        w = w - lr * dL_by_dw
        b = b - lr * dL_by_db
    return w.tolist(), float(b)