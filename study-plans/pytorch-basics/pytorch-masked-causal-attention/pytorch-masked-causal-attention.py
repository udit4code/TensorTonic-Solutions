import math
import torch

def causal_attention(Q, K, V):
    """
    Computes scaled dot-product causal attention.

    Args:
        Q: Query tensor of shape (..., seq_q, d_k)
        K: Key tensor of shape (..., seq_k, d_k)
        V: Value tensor of shape (..., seq_k, d_v)

    Returns:
        Attention output tensor of shape (..., seq_q, d_v)
    """
    # Step 1 : Compute pairwise similarity between every Query and every Key.
    # K.transpose(-2, -1) transposes only the last two dimensions.
    # Example:
    # Q: (..., seq_q, d_k)
    # K: (..., seq_k, d_k)
    # K.transpose(-2,-1): (..., d_k, seq_k)
    # Result: (..., seq_q, seq_k)
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)
    # Step 2 : Scale the attention scores.
    # Without scaling, the magnitude of the dot products grows with
    # d_k, causing softmax to become extremely peaked and producing
    # very small gradients during training.
    scores = scores / math.sqrt(d_k)
    # Step 3 : Create the causal mask.
    # We intentionally obtain the dimensions from the attention score matrix rather than from Q.
    # Why? scores always has shape (..., seq_q, seq_k)
    # making this implementation work for both:
    # - Self-attention (seq_q == seq_k)
    # - Cross-attention (seq_q != seq_k)
    seq_q = scores.shape[-2]
    seq_k = scores.shape[-1]
    # Step 4 :  Build a boolean upper-triangular mask.
    # Example (seq_len = 4):
    # False False True  True
    # False False False True
    # False False False False
    # False False False False
    # True means: "This position represents a future token and must not be attended to."
    # Why use a boolean mask instead of storing -inf?
    # - Consumes less memory.
    # - More expressive: the mask only describes *which* positions are invalid.
    # - Lets masked_fill() write -inf directly into the score matrix.
    mask = torch.triu(
        torch.ones(
            seq_q,
            seq_k,
            dtype=torch.bool,
            device=Q.device
        ),
        diagonal=1
    )

    # Step 5 : Replace all masked attention scores with -inf.
    # After softmax:
    # exp(-inf) = 0
    # therefore future tokens receive exactly zero attention probability.
    # Why is this device-safe?
    # Notice that the mask is created using: device=Q.device
    # This guarantees that both 'scores' and 'mask' live on the same device.
    # If Q is on CPU, then the mask is also on CPU.
    # If Q is on CUDA, then the mask is automatically created on CUDA.
    # Without specifying device=Q.device, PyTorch would create the
    # mask on the CPU by default, leading to runtime errors such as:
    # "Expected all tensors to be on the same device."

    scores = scores.masked_fill(
        mask,
        float("-inf")
    )

    # Step 6 : Convert attention scores into probabilities.
    # Softmax is applied along the key dimension so that each query
    # attends to all valid keys with probabilities summing to one.

    attention = torch.softmax(scores,dim=-1)
    # Step 7 : Compute the weighted sum of the Value vectors.
    # Every output vector becomes a weighted combination of the Value vectors, 
    # where the weights are the attention probabilities computed above.
    output = attention @ V
    return output