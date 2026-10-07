import torch

import torch
import torch.nn.functional as F

def kv_cache_decode_step(
    query: torch.Tensor,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
) -> dict[str, torch.Tensor]:
    """
    Computes a single grouped-query attention (GQA) decode step and returns
    the updated KV caches.

    Tensor shapes:
        query:       (B, Hq, D)
        new_key:     (B, Hkv, D)
        new_value:   (B, Hkv, D)
        key_cache:   (B, Hkv, S, D)
        value_cache: (B, Hkv, S, D)

    Returns dict containing:
        'output':          (B, Hq, D)
        'new_key_cache':   (B, Hkv, S + 1, D)
        'new_value_cache': (B, Hkv, S + 1, D)
    """
    B, Hq, D = query.shape
    _, Hkv, S, _ = key_cache.shape
    G = num_query_heads // num_kv_heads

    # Step 1 : Extend KV caches along the sequence dimension S: (B, Hkv, S + 1, D)
    new_key_cache = torch.cat([key_cache, new_key.unsqueeze(2)], dim=2)
    new_value_cache = torch.cat([value_cache, new_value.unsqueeze(2)], dim=2)

    # Step 2 : Expand KV heads to match query heads: (B, Hkv, S + 1, D) -> (B, Hq, S + 1, D)
    keys_expanded = (
        new_key_cache.unsqueeze(2)
        .expand(B, Hkv, G, S + 1, D)
        .reshape(B, Hq, S + 1, D)
    )
    values_expanded = (
        new_value_cache.unsqueeze(2)
        .expand(B, Hkv, G, S + 1, D)
        .reshape(B, Hq, S + 1, D)
    )

    # Step 3 : Add sequence dimension to query: (B, Hq, D) -> (B, Hq, 1, D)
    q = query.unsqueeze(2)

    # Step 4 : Scaled dot-product attention: (B, Hq, 1, D)
    attn_output = F.scaled_dot_product_attention(q, keys_expanded, values_expanded)

    # Step 5 : Squeeze sequence dimension back to (B, Hq, D)
    output = attn_output.squeeze(2)

    return {
        "output": output,
        "new_key_cache": new_key_cache,
        "new_value_cache": new_value_cache,
    }
