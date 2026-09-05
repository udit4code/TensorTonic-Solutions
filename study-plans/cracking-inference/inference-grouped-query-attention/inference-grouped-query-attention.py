import torch


def repeat_kv_from_scratch(x: torch.Tensor,num_query_heads: int) -> torch.Tensor:
    """
        x: (B, num_kv_heads, T, d_h)
    
        Returns:
            (B, num_query_heads, T, d_h)
    """
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    assert num_kv_heads > 0, f"num_kv_heads : {num_kv_heads}"
    assert num_query_heads % num_kv_heads == 0  , f"num_query_heads {num_query_heads} not divisible by num_kv_heads {num_kv_heads}"

    queries_per_kv_head = num_query_heads // num_kv_heads
    output = torch.empty(batch_size, num_query_heads, seq_len, head_dim, device=x.device, dtype=x.dtype)

    for query_head in range(num_query_heads):
        kv_head = query_head // queries_per_kv_head
        output[:, query_head, :, :] = x[:, kv_head, :, :]

    return output
    
def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    # Grouped Query Attention (GQA) is the middle ground between Multi-Head-Attention and Multi-Query-Attention. 
    # We still have many query heads, but instead of giving every query head its own K/V head like MHA, or forcing all query heads to share one K/V head like MQA, 
    # we divide the query heads into groups. Each group shares one K head and one V head.
    # The tradeoff is : GQA reduces KV-cache memory and decoding bandwidth substantially versus MHA while usually preserving better model quality than MQA. 

    # Step 0 : Extract  batch_size, seq_len, d_model from hidden_states, which is basically X. 
    # num_kv_heads is effectively the number of query-head groups.
    batch_size, seq_len, d_model = hidden_states.shape
    if d_model % num_query_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_query_heads ({num_query_heads})")

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

    # head_dim points to d_h
    head_dim = d_model // num_query_heads
    # Basically, no of queries in a group that share the same KV head
    queries_per_kv_head = num_query_heads // num_kv_heads


    # Step 1 : Project Q, K, V 
    # For Q, the dimension flows as : (B, T, D) @ (D, D) = (B, T, D)
    Q = hidden_states @ w_q
    # K has num_kv_heads * head_dim output dimensions
    # (B, T, D) @ (D, num_kv_heads * head_dim) -> (B, T, num_kv_heads * head_dim)
    K = hidden_states @ w_k
    # V has num_kv_heads * head_dim output dimensions
    # (B, T, D) @ (D, num_kv_heads * head_dim) -> (B, T, num_kv_heads * head_dim)
    V = hidden_states @ w_v

    # Step 2 : Split Q, K, V into heads 
    # Q's shape : (B, T, D) -> (B, T, H_q, d_h) via reshape(...) -> (B, H_q, T, d_h) via transpose(1, 2)
    Q = Q.reshape(batch_size, seq_len, num_query_heads, head_dim).transpose(1, 2)
    # K's shape : (B, T, num_kv_heads * d_h) -> (B, T, H_kv, d_h) via reshape -> (B, H_kv, T, d_h) via transpose(1, 2)
    K = K.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    # V's shape : (B, T, num_kv_heads * d_h) -> (B, T, H_kv, d_h) via reshape -> (B, H_kv, T, d_h) via transpose(1, 2)
    V = V.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Step 3 : Expand each K-V head across its query-head group.
    # In GQA, all query-heads within a group share the same K-V head for that group.
    # For Example: H_q = 8, H_kv = 2
    # K1, V1 -> Q1 Q2 Q3 Q4
    # K2, V2 -> Q5 Q6 Q7 Q8
    # We need to do (B, H_kv, T, Dh) -> (B, H_q, T, Dh) for K and V tensors.
    # How ? Via repeat_interleave, that literally duplicates tensor slices along a chosen dimension (In this case, dim=1).
    # Eg : K.shape = (B, 2, T, d_h) initially and along the head-dimension, K = [K0, K1]. 
    # After K = K.repeat_interleave(4, dim=1), along the head-dimension, K becomes [K0, K0, K0, K0, K1, K1, K1, K1]. 
    # For GQA, this makes each KV head line up with the query heads in its group.
    # Conceptually, PyTorch is doing something like this : 
    # result = []
    # for kv_head in K along dim=1:
    #     for _ in range(queries_per_kv_head):
    #         result.append(kv_head)
    # K = K.repeat_interleave(queries_per_kv_head,dim=1)
    K = repeat_kv_from_scratch(K, num_query_heads)
    # V = V.repeat_interleave(queries_per_kv_head,dim=1)
    V = repeat_kv_from_scratch(V, num_query_heads)

    # Step 4: Compute Attention scores
    # Shape of scores = (B, H_q, T, d_h) @ (B, H_q, d_h, T) = (B, H_q, T, T)
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    # Step 5: Apply Causal Mask, if enabled
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

    # Step 6 : Apply Softmax to convert attention scores into probabilities. 
    # Shape of attention_weights is still (B, H_q, T, T) as softmax doesn't change shape.
    attention_weights = torch.softmax(scores, dim=-1)

    # Step 7 : Get Weighted sum of values
    # Shape of head_outputs = (B, H_q, T, T) @ (B, H_q, T, d_h) = (B, H_q, T, d_h)
    head_outputs = attention_weights @ V

    # Step 8 : Concatenate the query head outputs 
    # Shape evolution = (B, H_q, T, d_h) -> (B, T, H_q, d_h) via transpose -> apply contiguous() -> (B, T, D) via reshape(...)
    concatenated_head_outputs = head_outputs.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)

    # Step 9 : Final Projection of concatenated_head_outputs
    # Shape = (B, T, D) @ (D, D) = (B, T, D)
    output = concatenated_head_outputs @ w_o
    return output


