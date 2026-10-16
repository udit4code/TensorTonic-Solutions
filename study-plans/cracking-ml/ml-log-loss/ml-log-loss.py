import numpy as np

def log_loss(y_true: list, y_pred: list) -> float:
    """
    Returns the mean binary log loss.
    """
    n = len(y_true)
    assert len(y_pred) == len(y_true) , "y_pred and y_true have different length"
    # Convert y_pred and y_true to numpy tensors
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    # Step 1 : Clip probabilities within the specified range 
    prob = np.clip(y_pred, 1e-15, 1.0 - 1e-15)

    # Step 2 : Compute the binary cross-entropy loss element-wise
    # Formula: - (y * log(p) + (1 - y) * log(1 - p))
    loss_elements = -(y_true * np.log(prob) + (1.0 - y_true) * np.log(1.0 - prob))

    # Step 3 : Compute the mean loss across all samples and return as a Python float
    total_loss = np.sum(loss_elements, axis=-1, keepdims=True) 
    mean_loss = total_loss / n 
    
    return round(float(mean_loss.item()), 4)
    
