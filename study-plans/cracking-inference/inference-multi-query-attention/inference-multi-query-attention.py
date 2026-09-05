import torch

def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    # The difference between Multi-Query-Attention and Multi-Head-Attention is that 
    # unlike in Multi-Head-Attention where each head has its own Q, K, V projections, in Multi-Query-Attention, each head has its own Query projection, but share the same K and V. 
    # The tradeoff is that : MQA has less representational power than MHA, but the size of K-V decreases significantly due to sharing of same K and V by all heads. 
    # Step 0 : Extract batch_size, seq_len, d_model from hidden_states
    batch_size, seq_len, d_model = hidden_states.shape
    assert d_model % num_query_heads == 0, f"d_model {d_model} is not divisible by num_query_heads {num_query_heads}"
    # head_dim is the head-width, or d_h. 
    head_dim = d_model // num_query_heads

    # Step 1 : Do Query Projection.  
    # Every query head gets its own representation.
    # (B, T, D) @ (D, D) -> (B, T, D)
    q = hidden_states @ w_q 

    # Step 2 : We split q into multiple query heads (given by num_query_heads)
    # (B, T, D) -> (B, T, H, d_h) via reshape(...) -> (B, H, T, Dh) via transpose(1, 2)
    q = q.reshape(batch_size, seq_len, num_query_heads, head_dim).transpose(1, 2)
    
    # Step 3 : Shared Key / Value projections
    # Unlike MHA, we create only ONE K head and ONE V head.
    # (B, T, D) @ (D, d_h) -> (B, T, d_h)
    k = hidden_states @ w_k
    v = hidden_states @ w_v

    # Step 4 : Add one head dimension to K and V 
    # (B, T, D) -> (B, T, 1, d_h) via reshape(...) -> (B, 1, T, Dh) via transpose(1, 2)
    k = k.reshape(batch_size, seq_len, 1, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, 1, head_dim).transpose(1, 2)

    # So, till now, we end up with the following situation : 
    # q: (B, H, T, d_h)
    # k: (B, 1, T, d_h)
    # v: (B, 1, T, d_h)

    # Step 5 : Compute scores and normalize the scores by a factor of sqrt(d_h)
    # q @ k.T -> (B, H, T, d_h) @ (B, 1, T, d_h).T = (B, H, T, d_h) @ (B, 1, d_h, T) = (B, H, T, T)
    # Under the hood, we do broadcasting of the head dimension as : 
    # H vs 1 -> k is shared across all query heads 
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    # Step 6: Apply causal mask if true
    if causal:
        # Why do we apply view ?
        # Because torch.arange(seq_len) gives a 1D tensor, but for the causal mask, 
        # we want one tensor to behave like a column and the other like a row so broadcasting creates a 2D matrix.
        # Say, seq_len = 4. Then, torch.arange(4) gives [0, 1, 2, 3] with shape = (4,). 
        # When we do .view(4, 1), it becomes [[0], [1], [2], [3]] for row_index. 
        # When we do .view(1, 4), it becomes [[0, 1, 2, 3]] for col_index. 
        row_index = torch.arange(seq_len, device=hidden_states.device).view(seq_len, 1)
        col_index = torch.arange(seq_len, device=hidden_states.device).view(1, seq_len)
        mask = row_index < col_index
        # In scores, at a given position say (i,j) for a given (B, H), if mask[i][j] = True, then, fill that cell with negative infinity, otherwise keep the same value as the initial scores
        scores = torch.where(mask, torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype), scores)

    # Step 7 : Convert Scores into [0,1] probabilities via softmax 
    # softmax doesn't alter shape of the tensor. So, shape of attention_weights is still (B, H, T, T).
    attention_weights = torch.softmax(scores, dim=-1)

    # Step 8 : Get the head outputs (i.e, the weighted sum using the same v head)
    # In MQA, all query heads share the same v head.
    # (B, H, T, T) @ (B, 1, T, d_h) = (B, H, T, d_h) via rules of broadcasting of the head-dimension
    head_outputs = attention_weights @ v 

    # Step 9 : Concatenate the head_outputs
    # (B, H, T, d_h) -> (B, T, H, Dh) via transpose -> (B, T, D) via reshape
    concatenated_head_outputs = head_outputs.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)

    # Step 10 : Output projection of the concatenated head outputs
    # (B, T, D) @ (D, D) -> (B, T, D) via batched matrix multiplication
    output = concatenated_head_outputs @ w_o

    return output