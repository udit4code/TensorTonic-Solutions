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
    true_class_probs = probs[np.arange(N), y_true]
    
    # Step 3 : Compute negative log-likelihood for each sample
    sample_losses = -np.log(true_class_probs)
    
    # Step 4 : Compute the mean loss across all samples and cast to a Python float
    total_loss = np.sum(sample_losses, axis=-1, keepdims=True)
    mean_loss = total_loss / N 
    
    return float(mean_loss.item())