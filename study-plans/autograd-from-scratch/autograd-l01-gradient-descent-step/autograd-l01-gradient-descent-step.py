import numpy as np

def gradient_descent_step(values: list, gradients: list, learning_rate: float) -> tuple[list, float]:
    """
    Returns a fresh list of updated values and the predicted objective change.
    """
    vals = np.asarray(values, dtype=float)
    grads = np.asarray(gradients, dtype=float)
    # Step 1 : Compute updated values: x_new = x - alpha * g
    updated_vals = vals - learning_rate * grads
    # Step 2 : Predicted objective change using first-order Taylor approximation: delta_f = g^T * delta_x
    step = -learning_rate * grads
    predicted_change = float(np.dot(grads, step))
    
    return updated_vals.tolist(), predicted_change