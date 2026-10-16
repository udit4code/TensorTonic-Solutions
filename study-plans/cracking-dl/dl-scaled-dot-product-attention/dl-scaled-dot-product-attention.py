import numpy as np

def attention_backward_pass(
    d_output: np.ndarray, 
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray, 
    attention_scores: np.ndarray, 
    d_k: int, 
    mask: np.ndarray = None
):
    d_out = np.asarray(d_output, dtype=np.float64)
        
    # Step 1 : Gradient w.r.t V and attention_scores (from Output = attention_scores @ V)
    dV = attention_scores.T @ d_out
    d_attention_scores = d_out @ V.T
    
    # Step 2 : Gradient w.r.t softmax input (d_scores)
    # Vectorized batch softmax derivative formula: d_softmax = S * (d_S - sum(d_S * S, axis=-1))
    sum_d_attn_times_attn = np.sum(d_attention_scores * attention_scores, axis=-1, keepdims=True)
    d_scores = attention_scores * (d_attention_scores - sum_d_attn_times_attn)
    
    # Step 3 : Gradient w.r.t mask step
    # If an element was masked out (-inf), its forward softmax score is 0. 
    # The multiplication by attention_scores above automatically zeroes out these gradients.
    if mask is not None:
        mask_bool = np.asarray(mask, dtype=bool)
        d_scores = np.where(mask_bool, d_scores, 0.0)
        
    # Step 4 : Gradient w.r.t scaling step (d_P)
    scale = np.sqrt(d_k)
    d_P = d_scores / scale
    
    # 5. Gradient w.r.t Q and K (from P = Q @ K.T)
    dQ = d_P @ K
    dK = d_P.T @ Q

    return dQ, dK, dV 

def attention_forward_pass(
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray, 
    d_k: int, 
    mask: np.ndarray = None
):
    # Step 1 : Compute P = Q@K.T / scale 
    P = Q @ K.T 
    scale = np.sqrt(d_k)
    scores = P / scale
    
    # Step 2 : Apply mask on P 
    if mask is not None:
        mask_bool = np.asarray(mask, dtype=bool)
        # Everywhere the mask is False (0), replace with -inf; otherwise keep original score
        scores = np.where(mask_bool, scores, -np.inf)
        
    # Step 3 : Apply softmax on P 
    row_wise_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - row_wise_max)
    sum_exp_scores = np.sum(exp_scores, axis=-1, keepdims=True)
    attention_scores = exp_scores / sum_exp_scores
    
    # Step 4 : Compute attention_scores @ V 
    output = attention_scores @ V 

    return output, attention_scores

    
def scaled_dot_product_attention(Q: list, K: list, V: list, mask: list | None = None, mode: str = 'forward', d_output: list | None = None) -> dict:
    """
    Returns forward results and optional backward gradients as a dictionary of lists.
    """
    # Convert Q, K, V into numpy tensors first
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d_k = Q.shape[-1]

    # By default, always do the forward pass .
    output, attention_scores = attention_forward_pass(Q, K, V, d_k, mask)
    result = {
        "output": [[round(float(v), 4) for v in row] for row in output],
        "attention_weights": [[round(float(v), 4) for v in row] for row in attention_scores]
    }
    
    # Optional Backward Pass
    if mode == "backward":
        assert d_output is not None, "d_output is None for backward mode"
        
        dQ, dK, dV = attention_backward_pass(d_output, Q, K, V, attention_scores, d_k, mask)
        
        result["dQ"] = [[round(float(v), 4) for v in row] for row in dQ]
        result["dK"] = [[round(float(v), 4) for v in row] for row in dK]
        result["dV"] = [[round(float(v), 4) for v in row] for row in dV]
        
    return result