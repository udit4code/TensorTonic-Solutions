import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Computes the InfoNCE loss for contrastive learning.
    Returns the loss as a float.
    """
    # Step 1: Convert inputs into numpy arrays and extract dimensions
    # Shape Evolution:
    #   Input Z1, Z2: lists of shape (N, d)
    #   After np.asarray: numpy ndarrays of shape (N, d)
    Z1 = np.asarray(Z1, dtype=np.float64)
    Z2 = np.asarray(Z2, dtype=np.float64)
    
    # N = batch size (number of independent examples)
    # d = feature dimension (size of the embedding vector per example)
    N, d = Z1.shape  
    
    # Ensure both representation matrices have identical dimensions
    assert Z1.shape == Z2.shape, f"Z1 and Z2 do not have the same shape"

    # Step 2: Compute pairwise dot-product logits (similarity scores)
    # Shape Evolution:
    #   Z1     : (N, d)
    #   Z2.T   : (d, N)  [Transposed so inner dimensions match for dot product]
    #   logits : (N, N)  [Resulting matrix of shape (N, N)]
    #
    # Intuition:
    # - Each entry (i, j) in the (N, N) logits matrix represents the similarity 
    #   (via dot product) between the i-th example in view Z1 and 
    #   the j-th example in view Z2.
    # - Diagonal elements (where i == j) represent the POSITIVE pairs: 
    #   the matching augmented views of the exact same underlying data point.
    # - Off-diagonal elements (where i != j) represent the NEGATIVE pairs: 
    #   comparisons between different data points in the batch.
    # - Division by 'temperature' scales the logits: a lower temperature 
    #   sharpens the probability distribution, making the model penalize 
    #   hard negatives more strongly.
    logits = (Z1 @ Z2.T) / temperature

    # Step 3: Compute Numerically Stable Cross-Entropy Loss (InfoNCE formulation)
    # Shape Evolution:
    #   logits       : (N, N)
    #   max_logits   : (N, 1) [Max value per row for numerical stability]
    #   exp_logits   : (N, N) [Exponentiated shifted logits]
    #   log_sum_exp  : (N, 1) [Log of the sum of exponentials along each row]
    #   pos_logits   : (N,)   [Extracted diagonal elements representing positive pairs]
    #   loss_per_row : (N,)   [Scalar loss per batch element]
    
    # Step 3.1 : Subtract max per row to prevent overflow during exponentiation (LogSumExp trick)
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    
    # Step 3.2 : Denominator of softmax in log space: log(sum(exp(logits)))
    log_sum_exp = max_logits + np.log(np.sum(exp_logits, axis=1, keepdims=True))
    
    # Step 3.3 : Numerator: Logits of the positive pairs lie on the diagonal (i == j)
    pos_logits = np.diagonal(logits)  # shape (N,)
    
    # Step 3.4 : InfoNCE loss for each row: -log(exp(pos) / sum(exp(all))) = -pos + log_sum_exp
    loss_per_row = -pos_logits + log_sum_exp.squeeze()
    
    # Step 4.1 : Average loss over the batch size N
    total_loss = np.mean(loss_per_row)
    
    return float(total_loss.item())