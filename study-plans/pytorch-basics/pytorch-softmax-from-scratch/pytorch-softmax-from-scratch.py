import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    # Convert input to a float tensor so exponentiation/division work reliably.
    logits = torch.as_tensor(logits, dtype=torch.float32)

    # Compute the maximum logit in each row.
    # Shape: (N, 1)
    # Used for the log-sum-exp trick to improve numerical stability.
    # torch.max() returns a named tuple: (values, indices), but we need only the values.
    row_max = torch.max(logits, dim=1, keepdim=True).values

    # Shift logits so the largest value in each row becomes 0.
    # Softmax is invariant to constant shifts: softmax(x) == softmax(x - c)
    shifted_logits = logits - row_max
    # Exponentiate the shifted logits.
    # Largest exponent is now exp(0)=1, avoiding overflow.
    exp_logits = torch.exp(shifted_logits)

    # Compute denominator: sum_j exp(logit_j)
    # Shape: (N, 1)
    row_sums = exp_logits.sum(dim=1, keepdim=True)

    # Normalize each row to obtain probabilities.
    # Every row now sums to 1.
    probabilities = exp_logits / row_sums

    return probabilities
