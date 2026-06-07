import numpy as np

# Training complexity: O(e⋅n⋅d)
# Space complexity: O(d)
# Where, n = number of samples, d = number of features, e = epochs
    
def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n_samples, n_features = X.shape

    # Initialize parameters
    W = np.zeros(n_features)
    b = 0.0
    # Training loop
    for _ in range(epochs):
        # Optimisation : We can remove this inner for loop across samples, via vectorization.
        for i in range(n_samples):
            # Linear score
            z = np.dot(X[i], W) + b
            # Step activation
            pred = 1 if z >= 0 else 0
            error = y[i] - pred
            # Perceptron update
            W += lr * error * X[i]
            b += lr * error

    return W.tolist(), float(b)
        
        
    