import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_k = Q.size(-1)

    # Step 1 : Compute raw-attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2 : Scale
    scores = scores / math.sqrt(d_k)

    # Step 3 : Softmax over keys
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4 : Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output