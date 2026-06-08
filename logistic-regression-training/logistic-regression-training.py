import numpy as np


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X, dtype=np.float64)
    N, d = X.shape
    y = np.array(y, dtype=np.float64)
    # Step 1 : Initialise weight and bias 
    b = 0.0
    w = np.zeros(d)
    # Step 2 : Now run training loop
    for step_id in range(steps):
        # Step 2.1 : Forward pass 
        z = X @ w + b 
        y_pred = 1.0 / (1 + np.exp(-z))
        # Step 2.2 : Backward pass 
        error = y_pred - y 
        dL_by_dw = (1.0 / N) * (X.T @ error)
        dL_by_db = (1.0 / N) * (np.sum(error))
        # Step 2.3 : Update weight and bias 
        w = w - lr * dL_by_dw
        b = b - lr * dL_by_db
    return w.tolist(), float(b)