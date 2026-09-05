import torch

def kv_cache_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_query_heads: int,
    gqa_kv_heads: int,
    head_dim: int,
    mla_latent_dim: int,
    mla_rotary_key_dim: int,
    bytes_per_element: int,
) -> torch.Tensor:
    """
    Returns:
        torch.int64 tensor of shape (4,)
        ordered as [MHA, MQA, GQA, MLA]
    """

    # MHA : Every query head has its own K head and V head.
    # Per token, per layer:
    #   K: num_query_heads * head_dim
    #   V: num_query_heads * head_dim
    # Total = 2 * num_query_heads * head_dim
    mha_bytes = batch_size * seq_len * num_layers * 2 * num_query_heads * head_dim * bytes_per_element
    

    # MQA : All query heads share exactly one K head and one V head.
    # Per token, per layer:
    #   K: head_dim
    #   V: head_dim
    # Total = 2 * head_dim
    mqa_bytes = batch_size * seq_len * num_layers * 2 * head_dim * bytes_per_element

    # GQA: There are gqa_kv_heads K heads and the same number of V heads.
    # Per token, per layer:
    #   K: gqa_kv_heads * head_dim
    #   V: gqa_kv_heads * head_dim
    # Total = 2 * gqa_kv_heads * head_dim
    gqa_bytes = batch_size * seq_len * num_layers * 2 * gqa_kv_heads * head_dim * bytes_per_element

    # MLA: Instead of caching full K and V tensors, cache:
    #   1. compressed KV latent representation
    #   2. rotary-position key component
    # Per token, per layer:
    #   mla_latent_dim + mla_rotary_key_dim
    # Note: there is NOT necessarily a factor of 2 here.
    
    mla_bytes = batch_size * seq_len * num_layers * (mla_latent_dim + mla_rotary_key_dim) * bytes_per_element

    return torch.tensor([mha_bytes, mqa_bytes, gqa_bytes, mla_bytes],dtype=torch.int64)
