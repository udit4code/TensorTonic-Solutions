import torch
from typing import Optional

def softmax_from_scratch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Step 1: subtract max for numerical stability
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    shifted = x - max_vals
    # Step 2: exponentiate
    exp_vals = torch.exp(shifted)
    # Step 3: normalize
    denominator = exp_vals.sum(dim=dim, keepdim=True)

    return exp_vals / denominator
    
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """
    d_k = query.shape[-1]
    # Step 1 : Compute product Q @ K.T and scale it by d_k 
    scores = query @ key.transpose(-2, -1) 
    factor = d_k ** 0.5 
    scores = scores / factor 
    # Step 2 : Add mask if available
    # mask is a Tensor with many boolean values. 
    # Python tries to reduce the whole Tensor to one True/False, but PyTorch refuses it because it is ambiguous. 
    # That is why, we use "if mask is not None" instead of "if mask". 
    if mask is not None:
        additive_mask = torch.zeros_like(scores)
        additive_mask[mask] = -float("inf")
        scores = scores + additive_mask
    # Step 3 : Compute Softmax to Attention weights
    attention_weights = softmax_from_scratch(scores, dim=-1)
    # Step 4 : Multiply Softmax with Value
    output = attention_weights @ value 
    return output
    
