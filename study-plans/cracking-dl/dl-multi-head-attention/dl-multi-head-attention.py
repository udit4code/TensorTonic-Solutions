import numpy as np

# We can think of Q, K, V as actually acting as the input_feature_matrices.
# In popular literature, they may be referred to as X_q, X_k, X_v (often in cross-attention, where the queries might come from a decoder and keys and values from an encoder).
def multi_head_attention(Q: list, K: list, V: list, W_q: list, W_k: list, W_v: list, W_o: list, num_heads: int, mask: list | None = None) -> dict:
    """
    Returns a dictionary of lists: output, attention_weights.
    """
    # Convert Q, K, V, W_q, W_k, W_v, W_o into tensors
    # Q, K, V have shape (T, d_model)
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    T, d_model = Q.shape
    # W_q, W_k, W_v, W_o have shapes (d_model, d_model)  
    W_q = np.asarray(W_q, dtype=np.float64)
    W_k = np.asarray(W_k, dtype=np.float64)
    W_v = np.asarray(W_v, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64) 
    
    # Step 1 : Get dimension of each head (Each head consists of unique key, value and query head in this case)
    assert d_model % num_heads == 0, f"d_model {d_model} is not divisible by num_heads {num_heads}"
    # d_h is dimension of each head.
    d_h = d_model // num_heads

    # Step 2 : Compute Linear Projections of Q, K, V
    # (T, d_model) @ (d_model, d_model) = (T, d_model) -> shape of Q_proj, K_proj, V_proj
    # Why .T ? Because in standard transformer implementations, weights are stored in (out_features, in_features)
    # orientation, meaning that the forward pass needs multiplication by the transpose Q @ W_q.T instead.
    Q_proj = Q @ W_q.T
    K_proj = K @ W_k.T 
    V_proj = V @ W_v.T 

    # Step 3 : Reshape each of Q_proj, K_proj and V_proj from (T, d_model) to (T, num_heads, d_h)
    Q_proj = Q_proj.reshape(T, num_heads, d_h)
    K_proj = K_proj.reshape(T, num_heads, d_h)
    V_proj = V_proj.reshape(T, num_heads, d_h)

    # Step 4 : Transpose each to shape (num_heads, T, d_h) from (T, num_heads, d_h)
    # Initially, (T, num_heads, d_h) = (0, 1, 2). We want it to be (1, 0, 2)
    Q_proj = Q_proj.transpose(1, 0, 2)
    K_proj = K_proj.transpose(1, 0, 2)
    V_proj = V_proj.transpose(1, 0, 2)
    # So far, shape of each Q_proj, K_proj, V_proj is (num_heads, T, d_h). 
    # We do this, because we want to compute attention for each head. 

    # Step 5 : Compute Scaled dot product for each head, but in 1 shot
    scale = np.sqrt(d_h)
    K_proj_transposed = K_proj.transpose(0, 2, 1)
    P = Q_proj @ K_proj_transposed / scale
    # Shape of P is (num_heads, T, T)

    # Step 5.1 : [Conditional] Apply mask on P, if provided
    if mask is not None:
        mask_np_arr = np.array(mask, dtype=bool)
        P = np.where(mask_np_arr, P, -np.inf)
        

    # Step 6 : Apply Stable Softmax on P
    row_max = np.max(P, axis=-1, keepdims=True)
    exp_scores = np.exp(P - row_max)
    sum_exp_scores = np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 7 : Compute attention weights
    # Shape of attention_weights is (num_heads, T, T)
    attention_weights = exp_scores / sum_exp_scores

    # Step 8 : Compute head outputs from attention_weights
    # shape of head_outputs is (num_heads, T, d_h)
    head_outputs = attention_weights @ V_proj

    # Shape 9 : Transpose and reshape back to (T, num_heads, d_h)
    # 9.1 Transpose back from (num_heads, T, d_h) to (T, num_heads, d_h)
    concat_heads = head_outputs.transpose(1, 0, 2)
    
    # 9.2 Reshape back to combine heads into (T, d_model)
    concat_heads = concat_heads.reshape(T, d_model)
    
    # Step 10 : Multiply by output projection matrix W_o -> Shape: (T, d_model)
    # Because of the orientation issue, we are using W_o.T 
    output = concat_heads @ W_o.T
    

    return {
        "output": [
            [
                round(float(v), 4) for v in row
            ] for row in output
        ],
        "attention_weights": [
            [
                [
                    round(float(v), 4) for v in row
                ] for row in hw
            ] for hw in attention_weights
        ]
    }
