import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    B, seq_len, d_model = q.shape 
    _, key_len, _ = k.shape 

    assert d_model % num_heads == 0 
    d_h = d_model // num_heads
    d_k = d_h 

    # Step 1 : Reshape q and k to separate heads
    q = q.reshape(B, seq_len, num_heads, d_h)
    k = k.reshape(B, key_len, num_heads, d_k)  

    # Step 2 : Transpose to move the num_heads dimension to the second position
    q = q.transpose(1, 2)  # Shape: (B, num_heads, seq_len, d_h)
    k = k.transpose(1, 2)  # Shape: (B, num_heads, key_len, d_k)

    # Step 3 : Compute scaled dot-product attention scores
    # q @ k^T results in shape (B, num_heads, seq_len, key_len)
    scale = torch.tensor(d_h, dtype=q.dtype).sqrt()
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    return scores
