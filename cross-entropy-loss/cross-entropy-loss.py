import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    N = len(y_true)
    
    # Convert inputs to NumPy arrays
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    
    # Step 1 : Clip probabilities for numerical stability (avoid log(0))
    probs = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    
    # Step 2 : Extract the predicted probability for the true class of each sample
    # Using advanced/fancy indexing: y_pred[row_indices, col_indices]
    # TRICK : 
    # Say, probs = np.array([
    #     [0.1, 0.7, 0.2],  # Sample 0: thinks it's class 1
    #     [0.8, 0.1, 0.1],  # Sample 1: thinks it's class 0
    #     [0.05, 0.05, 0.9] # Sample 2: thinks it's class 2
    # ])
    # Now, row_indices = np.arange(3) = [0, 1, 2]
    # And given, col_indices = [1, 0, 2]
    # Now, probs[row_indices, col_indices] = probs[[0, 1, 2], [1, 0, 2]] = Select cells (0, 1), (1, 0), (2, 2) = [0.7, 0.8, 0.9]
    row_indices = np.arange(N)
    col_indices = y_true
    true_class_probs = probs[row_indices, col_indices]
    # Instead of writing a slow Python for loop to manually extract [probs[0][y_true[0]], probs[1][y_true[1]], ...], advanced indexing extracts the probability of the exact correct class for every single sample in the dataset simultaneously in a single, lightning-fast C-level operation.
    
    # Step 3 : Compute negative log-likelihood for each sample
    sample_losses = -np.log(true_class_probs)
    
    # Step 4 : Compute the mean loss across all samples and cast to a Python float
    total_loss = np.sum(sample_losses, axis=-1, keepdims=True)
    mean_loss = total_loss / N 
    
    return float(mean_loss.item())