import torch
import math

def scaled_dot_product_attention(Q, K, V):
    """
    Args:
        Q: Query tensor of shape (..., seq_len_q, d_k)
        K: Key tensor of shape (..., seq_len_k, d_k)
        V: Value tensor of shape (..., seq_len_k, d_v)

    Returns:
        Attention output tensor of shape (..., seq_len_q, d_v)
    """
    # Step 1 :
    # Compute attention scores by measuring how similar every query is to every key.
    # Shape: (..., seq_len_q, seq_len_k)
    scores = Q @ K.transpose(-2, -1)
    # Step 2 :
    # Scale the scores by sqrt(d_k).
    # Without scaling, the dot products can become very large
    # when d_k is large, causing the softmax to become extremely
    # peaked and resulting in very small gradients.
    d_k = torch.tensor(Q.size(-1))
    scores = scores / torch.sqrt(d_k)
    # Step 3 : 
    # Convert the scores into attention weights.
    # Softmax is applied over the key dimension so that each
    # query assigns a probability distribution over all keys.
    attention_weights = torch.softmax(scores,dim=-1)
    # Step 4 : Compute a weighted sum of the value vectors.
    # Each output vector is a weighted combination of all values,
    # where the weights are given by the attention probabilities.
    output = attention_weights @ V
    return output