import numpy as np


def multi_head_attention(
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray, 
    W_q: np.ndarray, 
    W_k: np.ndarray, 
    W_v: np.ndarray, 
    W_o: np.ndarray, 
    num_heads: int, 
    mask: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Computes multi-head attention (reused for both self-attention and cross-attention)."""
    d_model = Q.shape[1]
    # Step 1 : Get dimension of each head.
    assert d_model % num_heads == 0, f"d_model {d_model} is not divisible by num_heads {num_heads}"
    d_h = d_model // num_heads
    
    # We assume that the weights are stored in (out_features, in_features) layout,
    # due to which the forward pass needs multiplication by W_q.T instead. 
    # Step 2 : Compute the projections via W_q, W_k, W_v 
    Q_proj = Q @ W_q.T 
    K_proj = K @ W_k.T 
    V_proj = V @ W_v.T 

    # Step 3 : Reshape each of Q_proj, K_proj, V_proj into (T_{i}, num_heads, d_h) from (T_{i}, d_model)
    # Reshape each projection using its own sequence length (handles cross-attention where T_q != T_k)
    T_q = Q.shape[0]
    T_k = K.shape[0]
    T_v = V.shape[0]
    
    Q_proj = Q_proj.reshape(T_q, num_heads, d_h)
    K_proj = K_proj.reshape(T_k, num_heads, d_h)
    V_proj = V_proj.reshape(T_v, num_heads, d_h)

    # Step 4 : Transpose each to shape (num_heads, T_{i}, d_h) from (T_{i}, num_heads, d_h)
    # So, we want to go to (1, 0, 2) from (0, 1, 2)
    Q_proj = Q_proj.transpose(1, 0, 2)
    K_proj = K_proj.transpose(1, 0, 2)
    V_proj = V_proj.transpose(1, 0, 2)
    # So far, each of Q_proj, K_proj, V_proj has shape (num_heads, T_{i}, d_h)

    # Step 5 : Comput scaled dot product for each head, using vectorization
    scale = np.sqrt(d_h)
    P = Q_proj @ K_proj.transpose(0, 2, 1)
    P = P / scale
    # Shape of P is (num_heads, T_q, T_k)

    # Step 6 : Apply conditional mask on P, if present
    if mask is not None:
        mask_arr = np.asarray(mask)
        # Broadcast mask across heads if shape is (T_q, T_k)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[np.newaxis, :, :]
        P = np.where(mask_arr, P, P - 1e9)

    # Step 7 : Apply Stable Softmax on P
    row_max = np.max(P, axis=-1, keepdims=True)
    exp_scores = np.exp(P - row_max)
    sum_exp_scores = np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 8 : Compute Attention weights
    # Shape of attention_weights is (num_heads, T_q, T_k)
    attention_weights = exp_scores / sum_exp_scores

    # Step 9 : Compute head_output from attention_weights
    # shape of head_outputs is (num_heads, T_q, d_h)
    head_outputs = attention_weights @ V_proj

    # Shape 10 : Transpose and reshape back to (T, num_heads, d_h)
    # 9.1 Transpose back from (num_heads, T_q, d_h) to (T_q, num_heads, d_h)
    concat_heads = head_outputs.transpose(1, 0, 2)
    
    # 9.2 Reshape back to combine heads into (T_q, d_model)
    concat_heads = concat_heads.reshape(T_q, d_model)
    
    # Step 10 : Multiply by output projection matrix W_o -> Shape: (T_q, d_model)
    # Because of the orientation issue, we are using W_o.T 
    output = concat_heads @ W_o.T

    return {
        "output" : output,
        "attention_weights" : attention_weights,
    }

    
def layer_normalization(
    x: np.ndarray, 
    gamma: np.ndarray, 
    beta: np.ndarray, 
    eps: float = 1e-5
) -> np.ndarray:
    """Applies layer normalization across the feature dimension."""
    # x has shape (T, d_model)
    # Step 1 : Compute the mean across the feature dimension (axis=-1)
    mean = np.mean(x, axis=-1, keepdims=True)
    # Step 2: Compute the variance across the feature dimension (axis=-1).
    variance = np.var(x, axis=-1, keepdims=True)
    # Step 3: Normalize the input (zero mean, unit variance)
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    # Step 4: Scale by learnable gamma and shift by learnable beta.
    output = x_normalized * gamma + beta

    return output


def feed_forward_network(
    x: np.ndarray, 
    W1: np.ndarray, 
    b1: np.ndarray, 
    W2: np.ndarray, 
    b2: np.ndarray
) -> np.ndarray:
    """Applies position-wise feed-forward network (Linear -> ReLU -> Linear)."""
    # Step 1: First linear transformation (expansion layer)
    linear1 = x @ W1.T + b1
        
    # Step 2: Apply ReLU non-linearity
    activated = np.maximum(0.0, linear1)
        
    # Step 3: Second linear transformation (projection layer back to d_model)
    output = activated @ W2.T + b2
        
    return output    


def decoder_block(
    x: list, 
    enc_output: list, 
    W_q1: list, W_k1: list, W_v1: list, W_o1: list, 
    W_q2: list, W_k2: list, W_v2: list, W_o2: list, 
    num_heads: int, 
    W1: list, b1: list, W2: list, b2: list, 
    gamma1: list, beta1: list, 
    gamma2: list, beta2: list, 
    gamma3: list, beta3: list, 
    mask: list | None = None
) -> dict:
    """
    Returns a dictionary of lists: self_attn, norm1, cross_attn, norm2, ffn_output, output.
    """
    # Step 1 : Convert inputs to NumPy arrays at the entry point
    x_arr = np.asarray(x, dtype=np.float64)
    enc_arr = np.asarray(enc_output, dtype=np.float64)
    
    W_q1 = np.asarray(W_q1, dtype=np.float64)
    W_k1 = np.asarray(W_k1, dtype=np.float64)
    W_v1 = np.asarray(W_v1, dtype=np.float64)
    W_o1 = np.asarray(W_o1, dtype=np.float64)
    
    W_q2 = np.asarray(W_q2, dtype=np.float64)
    W_k2 = np.asarray(W_k2, dtype=np.float64)
    W_v2 = np.asarray(W_v2, dtype=np.float64)
    W_o2 = np.asarray(W_o2, dtype=np.float64)
    
    W1 = np.asarray(W1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    
    gamma1 = np.asarray(gamma1, dtype=np.float64)
    beta1 = np.asarray(beta1, dtype=np.float64)
    gamma2 = np.asarray(gamma2, dtype=np.float64)
    beta2 = np.asarray(beta2, dtype=np.float64)
    gamma3 = np.asarray(gamma3, dtype=np.float64)
    beta3 = np.asarray(beta3, dtype=np.float64)
    
    # Step 2 : Sub-layer 1: Masked Multi-Head Self-Attention (Q=x, K=x, V=x)
    self_attn_dict = multi_head_attention(x_arr, x_arr, x_arr, W_q1, W_k1, W_v1, W_o1, num_heads, mask)
    self_attn_output = np.asarray(self_attn_dict["output"])
    
    # Step 3 : First Residual Connection + Layer Normalization
    norm1_input = x_arr + self_attn_output
    norm1_output = layer_normalization(norm1_input, gamma1, beta1)
    
    # Step 4 : Sub-layer 2: Multi-Head Cross-Attention (Q from decoder, K & V from encoder output)
    cross_attn_dict = multi_head_attention(norm1_output, enc_arr, enc_arr, W_q2, W_k2, W_v2, W_o2, num_heads, mask=None)
    cross_attn_output = np.asarray(cross_attn_dict["output"])
    
    # Step 5 : Second Residual Connection + Layer Normalization
    norm2_input = norm1_output + cross_attn_output
    norm2_output = layer_normalization(norm2_input, gamma2, beta2)
    
    # Step 6 : Sub-layer 3: Position-wise Feed-Forward Network
    ffn_output = feed_forward_network(norm2_output, W1, b1, W2, b2)
    
    # Step 7 : Third Residual Connection + Final Layer Normalization
    norm3_input = norm2_output + ffn_output
    final_output = layer_normalization(norm3_input, gamma3, beta3)

    format_transform = lambda arr: [[round(float(v), 4) for v in row] for row in arr]
    
    return {
        "self_attn": format_transform(self_attn_output.tolist()),
        "norm1": format_transform(norm1_output.tolist()),
        "cross_attn": format_transform(cross_attn_output.tolist()),
        "norm2": format_transform(norm2_output.tolist()),
        "ffn_output": format_transform(ffn_output.tolist()),
        "output": format_transform(final_output.tolist()),
    }