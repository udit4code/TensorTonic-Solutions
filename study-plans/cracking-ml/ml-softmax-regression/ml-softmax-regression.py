import numpy as np

def compute_softmax():
    pass 
    
def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    N, d = X.shape
    # Step 1 : Initialis w and b
    w = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    # Convert labels to one_hot vectors.
    # Eg : z = np.eye(2) = [[1, 0], [0, 1]] and y = [1, 1] and now z[y] = [[0, 1], [0, 1]]
    Y = np.eye(n_classes)[y]
    # Step 2 : Start Training loop
    for _ in range(n_iters):
        # Step 2.1 : Compute logits  (the raw value before applying softmax) from X.
        z = X @ w + b
        # Step 2.2 : stable softmax
        z_max = np.max(z,axis=1,keepdims=True)
        z = z - z_max
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        # Step 2.3 : Compute gradients
        diff = probs - Y
        dL_by_dw = (X.T @ diff) / N
        dL_by_db = np.sum(diff, axis=0) / N
        # update
        w -= lr * dL_by_dw
        b -= lr * dL_by_db

    return w.tolist(), b.tolist()