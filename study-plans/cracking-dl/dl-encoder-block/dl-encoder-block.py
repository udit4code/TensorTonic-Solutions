import numpy as np

def multi_head_attention(
    X: np.ndarray,
    W_q: np.ndarray, 
    W_k: np.ndarray, 
    W_v: np.ndarray, 
    W_o: np.ndarray, 
    num_heads: int
) -> dict[str, np.ndarray]:
    """
    Computes multi-head attention given NumPy ndarrays for Q, K, V and weights.
    Returns a dictionary with 'output' and 'attention_weights' as np.ndarrays.
    """
    T, d_model = X.shape
    # Step 1 : Get dimension of each head
    assert d_model % num_heads == 0 , f"d_model {d_model} is not divisible by {num_heads}"
    d_h = d_model // num_heads 

    # Step 2 : Project X onto W_q, W_k, W_v linear projections 
    Q = X @ W_q.T 
    K = X @ W_k.T
    V = X @ W_v.T 

    # Step 3 : Reshape each of Q, K, V from (T, d_model) to (T, num_heads, d_h)
    Q = Q.reshape(T, num_heads, d_h)
    K = K.reshape(T, num_heads, d_h)
    V = V.reshape(T, num_heads, d_h)

    # Step 4 : Transpose each to shape (num_heads, T, d_h) 
    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)

    # Step 5 : Compute scaled dot product for each head. 
    scale = np.sqrt(d_h)
    # We want K_transposed to be of shape (num_heads, d_h, T)
    K_transposed = K.transpose(0, 2, 1)
    P = (Q @ K_transposed) / scale

    # Step 6 : Apply Stable Softmax on P and compute attention_weights
    row_max = np.max(P, axis=-1, keepdims=True)
    exp_scores = np.exp(P - row_max)
    sum_exp_scores = np.sum(exp_scores, axis=-1, keepdims=True)
    # Shape of attention_weights is (num_heads, T, T)
    attention_weights = exp_scores / sum_exp_scores

    # Step 8 : Compute head outputs from attention_weights
    # shape of head_outputs is (num_heads, T, d_h)
    head_outputs = attention_weights @ V

    # Shape 9 : Transpose and reshape back to (T, num_heads, d_h)
    # 9.1 Transpose back from (num_heads, T, d_h) to (T, num_heads, d_h)
    concat_heads = head_outputs.transpose(1, 0, 2)
    
    # 9.2 Reshape back to combine heads into (T, d_model)
    concat_heads = concat_heads.reshape(T, d_model)
    
    # Step 10 : Multiply by output projection matrix W_o -> Shape: (T, d_model)
    # Because of the orientation issue, we are using W_o.T 
    output = concat_heads @ W_o.T

    return {
        "output" : output,
        "attention_weights" : attention_weights,
    }

    
def layer_normalization(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Applies Layer Normalization to the input tensor.
    
    Formula: LN(x) = (x - mean) / sqrt(variance + eps) * gamma + beta
    
    Args:
        x: Input tensor of shape (T, d_model)
        gamma: Learnable scale parameter of shape (d_model,)
        beta: Learnable shift parameter of shape (d_model,)
        eps: Small constant added to variance for numerical stability
        
    Returns:
        Normalized tensor of shape (T, d_model)
    """
    # Step 1 : Compute the mean across the feature dimension (axis = -1)
    # x is of shape (T, d_model)
    # keepdims=True ensures the shape becomes (T, 1) instead of (T,) so it broadcasts properly.
    mean = np.mean(x, axis=-1, keepdims=True)

    # Step 2: Compute the variance across the feature dimension (axis=-1).
    # Using keepdims=True keeps the shape as (T, 1) for safe broadcasting.
    variance = np.var(x, axis=-1, keepdims=True)

    # Step 3: Normalize the input (zero mean, unit variance) and add eps to the denominator to prevent division by zero.
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    
    # Step 4: Scale by learnable gamma and shift by learnable beta.
    # NumPy broadcasts gamma and beta of shape (d_model,) across the sequence dimension (T, d_model).
    output = x_normalized * gamma + beta

    return output

import numpy as np

def feed_forward_network(
    x: np.ndarray, 
    W1: np.ndarray, 
    b1: np.ndarray, 
    W2: np.ndarray, 
    b2: np.ndarray
) -> np.ndarray:
    """
    Applies Position-wise Feed-Forward Network (Linear -> ReLU -> Linear).
    
    Formula: FFN(x) = max(0, x @ W1.T + b1) @ W2.T + b2
    
    Args:
        x: Input tensor of shape (T, d_model)
        W1: First weight matrix of shape (d_ff, d_model) [PyTorch linear convention]
        b1: First bias vector of shape (d_ff,)
        W2: Second weight matrix of shape (d_model, d_ff)
        b2: Second bias vector of shape (d_model,)
        
    Returns:
        Output tensor of shape (T, d_model)
    """
    # Step 1: First linear transformation (expansion layer)
    # Shape: (T, d_model) @ (d_model, d_ff) + (d_ff,) -> (T, d_ff)
    linear1 = x @ W1.T + b1
    
    # Step 2: Apply ReLU non-linearity
    activated = np.maximum(0.0, linear1)
    
    # Step 3: Second linear transformation (projection layer back to d_model)
    # Shape: (T, d_ff) @ (d_ff, d_model) + (d_model,) -> (T, d_model)
    output = activated @ W2.T + b2
    
    return output

    
def encoder_block(x: list, W_q: list, W_k: list, W_v: list, W_o: list, num_heads: int, W1: list, b1: list, W2: list, b2: list, gamma1: list, beta1: list, gamma2: list, beta2: list) -> dict:
    """
    Returns a dictionary of lists: attention_output, norm1, ffn_output, output.
    """
    # Convert x, W_q, W_k, W_v, W_o, W1, W2 to numpy ndarrays
    x = np.asarray(x, dtype=np.float64)
    W_q = np.asarray(W_q, dtype=np.float64)
    W_k = np.asarray(W_k, dtype=np.float64)
    W_v = np.asarray(W_v, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    beta1 = np.asarray(beta1, dtype=np.float64)
    gamma1 = np.asarray(gamma1, dtype=np.float64)
    beta2 = np.asarray(beta2, dtype=np.float64)
    gamma2 = np.asarray(gamma2, dtype=np.float64)
    

    # Step 1 : Multi-Head Self-Attention (Q, K, V all come from x)
    attn_dict = multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads)
    attn_output = np.asarray(attn_dict["output"])
    
    # 2. First Residual Connection & Layer Norm (Post-LN or Pre-LN pattern)
    # Example (Post-LN style): Norm(x + Attention_Output)
    x_norm1 = layer_normalization(x + attn_output, gamma1, beta1)
    
    # 3. Feed-Forward Network
    ffn_output = feed_forward_network(x_norm1, W1, b1, W2, b2)
    
    # 4. Second Residual Connection & Layer Norm
    output = layer_normalization(x_norm1 + ffn_output, gamma2, beta2)


    output_transform = lambda arr: [[round(float(v), 4) for v in row] for row in arr] 
    return {
        "attention_output": output_transform(attn_output.tolist()),
        "norm1": output_transform(x_norm1.tolist()),
        "ffn_output": output_transform(ffn_output.tolist()),
        "output": output_transform(output.tolist()),
    }
