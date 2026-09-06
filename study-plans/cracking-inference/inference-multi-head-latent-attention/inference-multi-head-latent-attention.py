import torch
from typing import Tuple

# Multi-Head Latent Attention is an optimization introduced by Deepseek to reduce
# the KV Cache size during inference, while retaining some of the benefits of Multi-Head-Attention. 
# The core idea is: instead of caching a separate full-sized Key and Value representation for every token, 
# MLA first compresses them into a much smaller latent vector, caches only that latent vector, 
# and reconstructs the per-head K/V information when attention is computed.
# MLA is clever because it doesn't simply force heads to share identical K/V representations. 
# Instead, different heads can obtain different projections from the same compressed latent.


def multi_head_latent_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_down: torch.Tensor,
    w_up_k: torch.Tensor,
    w_up_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (output tensor of shape (batch, seq, d_model), latent tensor of shape (batch, seq, d_latent))
    """
    # Step 0 : Extract batch_size, seq_len, d_model from hiddent_states
    batch_size, seq_len, d_model = hidden_states.shape
    assert d_model % num_heads == 0
    # head_dim is d_h 
    head_dim = d_model // num_heads
    # Step 1 : Queries are projected normally.
    # (B, S, d_model) @ (d_model, d_model) -> (B, S, d_model)
    q = hidden_states @ w_q
    # Step 2 : Compress hidden states into the KV latent space.
    # Instead of storing full K and V, MLA stores this smaller representation.
    # (B, S, d_model) @ (d_model, d_latent) -> (B, S, d_latent)
    _, d_latent = w_down.shape 
    latent = hidden_states @ w_down
    # Step 3 :  Reconstruct K and V from the latent representation.
    # (B, S, d_latent) @ (d_latent, d_model) -> (B, S, d_model)
    k = latent @ w_up_k
    v = latent @ w_up_v
    # Step 4 : Split Q/K/V into heads.
    # (B, S, d_model) -> (B, S, H, d_h) via reshape() -> (B, H, S, d_h) via transpose
    q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    # Step 5 : Compute Attention scores via Q @ K^T 
    # (B,H,S,d_h) @ (B,H,d_h,S) -> (B,H,S,S)
    scores = torch.matmul(q, k.transpose(-2, -1))
    # Step 6 : Normalize attention scores by sqrt(d_h)
    scores = scores / math.sqrt(head_dim)
    # Step 7 : Apply Causal mask if true
    if causal: 
        row_idx = torch.arange(seq_len, device=hidden_states.device).view(seq_len, 1)
        col_idx = torch.arange(seq_len, device=hidden_states.device).view(1, seq_len)
        mask = row_idx < col_idx
        scores = torch.where(mask, torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype), scores)
    # Step 8 : Apply Softmax on the attention-scores to convert them to probabilities
    attention_weights = torch.softmax(scores, dim=-1)
    # Step 9 : Get Weighted combination of values.
    # (B,H,S,S) @ (B,H,S,d_h) -> (B,H,S, d_h)
    context = attention_weights @ v
    # Step 9 : Merge heads.
    # (B,H,S, d_h) -> (B,S, d_model)
    context = context.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)

    output = context @ w_o
    return output, latent
    
    
