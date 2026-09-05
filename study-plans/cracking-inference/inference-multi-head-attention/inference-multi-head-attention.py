import torch

def safe_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax from first principles in a numerically stable way.
    """
    # Step 1 : Find maximum value along the softmax dimension
    max_x = torch.max(x, dim=dim, keepdim=True).values
    # Step 2 : Shift values so the largest value becomes 0
    shifted_x = x - max_x
    # Step 3 : Exponentiate
    exp_x = torch.exp(shifted_x)
    # Step 4 : Sum exponentials
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    # Step 5 : Normalize
    return exp_x / sum_exp_x
    
def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    # Step 0 : Extract batch_size, seq_len, d_model from hidden_states
    batch_size, seq_len, d_model = hidden_states.shape
    assert d_model % num_heads == 0, f"d_model {d_model} is not divisible by num_heads {num_heads}"
    head_dim = d_model // num_heads
    
    # Step 1 : We do linear projection and get Q, K, V matrices
    # hidden_states can be thought of as X. 
    # So, for Q, we have Q = X @ w_q . X is (B, T, D) and w_q is (D, D). 
    # Hence, (B, T, D) @ (D, D) = (B, T, D).
    # PyTorch’s @ does batched matrix multiplication.
    # PyTorch treats the last two dimensions of X as a matrix: (T,D)
    # and applies the same weight matrix w_q of shape (D, D) to every batch item.
    q = hidden_states @ w_q
    k = hidden_states @ w_k
    v = hidden_states @ w_v

    # Step 2 :  Split d_model into multiple heads (as given by num_heads)
    # (B, T, D) -> (B, T, H, d_h) via  q.reshape(batch_size, seq_len, num_heads, head_dim)  -> (B, H, T, d_h) via transpose(1, 2)
    # At the end of this step,  q, k, v will end up with shape : (B, H, T, d_h), where H means num_heads
    q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Step 3 : Compute Attention-scores
    # For q @ k^T (For transpose, we do (-2, -1)), the dimensions flow as : 
    # (B, H, T, d_h) @ (B, H, d_h, T) -> (B, H, T, T)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    # Step 4 : Apply optional causal mask, if present 
    if causal:
        # Say, seq_len = 4. Then, row_idx = [[0], [1], [2], [3]] and col_idx = [0, 1, 2, 3]
        row_idx = torch.arange(seq_len, device=hidden_states.device).view(seq_len, 1)
        col_idx = torch.arange(seq_len, device=hidden_states.device).view(1, seq_len)
        # True wherever key position j is in the future of query position i
        # mask[i,j]= (j>i) because query token i must not attend to a future key token j. 
        #                 col_idx
        #         0      1      2      3
        # row 0   F      T      T      T
        # row 1   F      F      T      T
        # row 2   F      F      F      T
        # row 3   F      F      F      F
        # Hence, PyTorch broadcasts them to a (4, 4) comparison. 
        mask = col_idx > row_idx
        # torch.where(mask, -inf, scores) means: wherever the mask is True, replace that attention score with -inf; otherwise keep the original score. 
        # After softmax, those -inf positions get probability 0
        scores = torch.where(
            mask,
            torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype),
            scores
        )

    # Step 5: Convert these scores into probabilities via softmax 
    # alternate approach : torch.softmax(scores, dim=-1)
    attention_weights = safe_softmax(scores, dim=-1)

    # Step 6: Weighted sum of values
    head_outputs = attention_weights @ v 

    # Step 7 : Concatenate heads
    # (B, H, T, d_h) -> (B, T, H, d_h) -> (B, T, D)
    concatenated = head_outputs.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)

    # Step 8 : Final output projection
    # (B, T, D) @ (D, D) -> (B, T, D)
    output = concatenated @ w_o

    return output

    
    

    

